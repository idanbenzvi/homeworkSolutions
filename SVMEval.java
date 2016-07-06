package hw5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SVMEval {
	public SMO m_smo;
	public ArrayList<Integer> indicesToRemove;

	private String M_EMPTY_REL = "empty_ElectionsData.txt";

	public SVMEval() {
		this.m_smo = new SMO();
		indicesToRemove = new ArrayList<Integer>();
	}

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public double crossValidation(Instances instances, int numFolds) throws Exception {
		Random random = new Random();
		instances.randomize(random);
		double numInstancesPerFold = ((double) instances.numInstances()) / numFolds;

		Instances emptyData = new Instances(instances);
		emptyData.delete();
		emptyData.setClassIndex(instances.classIndex());

		int curCutOff = (int) Math.floor(numInstancesPerFold);
		Instances[] folds = new Instances[numFolds];
		loadFolds(instances, numFolds, emptyData, curCutOff, folds);

		Instances training = new Instances(emptyData);
		Instances test = new Instances(emptyData);
		double errorSum = 0;
		long sumTime = 0;
		for (int i = 0; i < numFolds; i++) {
			loadTestAndTraining(numFolds, folds, training, test, i);
			m_smo.buildClassifier(training);
			long curTime = System.nanoTime();
			errorSum += calcAvgError(test);
			long estTime = System.nanoTime() - curTime;
			sumTime += estTime;
			training.delete();
			test.delete();
		}
		long estTime = sumTime / numFolds;
		// System.out.println("took " + estTime + " nano seconds");
		double crossValidError = ((double) errorSum) / numFolds;
		return crossValidError;
	}

	private void loadTestAndTraining(int numFolds, Instances[] folds, Instances training, Instances test, int i) {
		for (int k = 0; k < folds[i].numInstances(); k++) {
			test.add(folds[i].instance(k));
		}
		for (int j = 0; j < numFolds; j++) {
			if (j != i) {
				for (int k = 0; k < folds[j].numInstances(); k++) {
					training.add(folds[j].instance(k));
				}
			}
		}
	}

	private void loadFolds(Instances instances, int numFolds, Instances emptyData, int curCutOff, Instances[] folds) {
		for (int i = 0; i < numFolds; i++) {
			folds[i] = new Instances(emptyData);
		}
		int curIndex = 0;
		for (int i = 0; i < numFolds - 1; i++) {
			for (int j = 0; j < curCutOff; j++) {
				folds[i].add(instances.instance(curIndex));
				curIndex++;
			}
		}
		int lastFold = numFolds - 1;
		for (int j = curIndex; j < instances.numInstances(); j++) {
			folds[lastFold].add(instances.instance(j));
		}
	}

	public double calcAvgError(Instances instances) throws Exception {
		double sumError = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance curInstance = instances.instance(i);
			boolean isDifferent = m_smo.classifyInstance(curInstance) == curInstance.classValue() ? false : true;
			if (isDifferent) {
				sumError += 1;
			}
		}

		double avgError = sumError / instances.numInstances();
		return avgError;
	}

	private void setGamma(RBFKernel kernel, Instances instances) throws Exception {
		double gamma;
		double minError = 1;
		double minGamma = Math.pow(2, -15);
		for (int i = -15; i < 2; i++) {
			gamma = Math.pow(2, i);
			kernel.setGamma(gamma);
			m_smo.setKernel(kernel);
			double curError = crossValidation(instances, 3);
			if (curError < minError) {
				minError = curError;
				minGamma = gamma;
			}
		}
		kernel.setGamma(minGamma);
	}

	public void chooseKernel(Instances instances) throws Exception {
		double minError = 1;
		Kernel bestKernel = new RBFKernel();
		RBFKernel kernel2 = new RBFKernel();
		setGamma(kernel2, instances);
		m_smo.setKernel(kernel2);
		double error = crossValidation(instances, 3);
		if (error < minError) {
			minError = error;
			bestKernel = kernel2;
		}
		for (int i = 2; i < 4; i++) {
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(i);
			m_smo.setKernel(kernel);
			error = crossValidation(instances, 3);
			if (error < minError) {
				minError = error;
				bestKernel = kernel;
			}
		}

		m_smo.setKernel(bestKernel);
	}

	public Instances backwardsWrapper(Instances instances, double threshold, int k) throws Exception {
		Instances curInstances = instances;
		int numIterations = 0;
		int numAttributes = curInstances.numAttributes();
		System.out.println("Num attr is " + numAttributes);
		double originalError = crossValidation(curInstances, 3);
		double errorDiff = 0;
		Remove remove;
		while (numAttributes > k && errorDiff < threshold) {
			double minError = 1;
			int minI = 0;
			for (int i = 0; i < numAttributes; i++) {
				if (i != curInstances.classIndex()) {
					remove = new Remove();
					remove.setInputFormat(curInstances);

					String[] options = new String[2];
					options[0] = "-R";
					options[1] = Integer.toString(i + 1);
					remove.setOptions(options);
					Instances workingSet = Filter.useFilter(curInstances, remove);
//					System.out.println(workingSet.numAttributes());
					double curError = crossValidation(workingSet, 3);
					if (curError < minError) {
						minError = curError;
						minI = i;
					}
					numIterations += 1;
				}
			}
			remove = new Remove();
			remove.setInputFormat(curInstances);
			String[] options = new String[2];
			options[0] = "-R";
			options[1] = Integer.toString(minI + 1);
			remove.setOptions(options);
			Instances workingSet = Filter.useFilter(curInstances, remove);
			double curError = crossValidation(workingSet, 3);
			errorDiff = curError - originalError;
			if (errorDiff < threshold) {
				indicesToRemove.add(minI);
				curInstances = workingSet;
				numAttributes = curInstances.numAttributes();
			}
			System.out.println("Num iterations is " + numIterations + " num attributes is " + numAttributes + " and the error dif is " + errorDiff);
		}
		return curInstances;
	}

	public Instances removeNonSelectedFeatures(Instances instances) throws Exception {
		Remove remove;
		Instances curInstances = new Instances(instances);
		for (int index : indicesToRemove) {
			remove = new Remove();
			remove.setInputFormat(curInstances);

			String[] options = new String[2];
			options[0] = "-R";
			options[1] = Integer.toString(index + 1);
			remove.setOptions(options);
			Instances workingSet = Filter.useFilter(curInstances, remove);
			curInstances = workingSet;
		}
		return curInstances;
	}
	
	public void buildClassifier(Instances instances) throws Exception{
		m_smo.buildClassifier(instances);
	}

	public static void main(String[] args) throws Exception {
		String training = "ElectionsData_train.txt";
		String testing = "ElectionsData_test.txt";
		BufferedReader datafile = readDataFile(training);

		Instances data = new Instances(datafile);

		data.setClassIndex(0);
		SVMEval eval = new SVMEval();
		eval.chooseKernel(data);
		Instances workingSet = eval.backwardsWrapper(data, 0.05, 5);

		eval.buildClassifier(workingSet);

		BufferedReader datafile2 = readDataFile(testing);
		Instances dataTest = new Instances(datafile2);
		dataTest.setClassIndex(0);
		Instances subsetOfFeatures = eval.removeNonSelectedFeatures(dataTest);

		double avgError = eval.calcAvgError(subsetOfFeatures);

		System.out.println(avgError);
	}

}
