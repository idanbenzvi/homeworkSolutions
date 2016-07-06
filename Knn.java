package hw4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class Knn implements Classifier {
	Instances m_trainingInstances;
	int m_k = 1;
	int m_power = 2;
	String m_distanceType = "";

	private String M_EMPTY_REL = "empty_ElectionsData.txt";
	
	private String M_MODE = "";

	public String getM_MODE() {
		return M_MODE;
	}

	public void setM_MODE(String m_MODE) {
		M_MODE = m_MODE;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		switch (M_MODE){
		case "none":
			noEdit(arg0);
			break;
		case "forward":
			editedForward(arg0);
			break;
		case "backward":
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
	}

	public String getM_distanceType() {
		return m_distanceType;
	}

	public void setM_distanceType(String m_distanceType) {
		this.m_distanceType = m_distanceType;
	}

	public int getM_power() {
		return m_power;
	}

	public void setM_power(int m_power) {
		this.m_power = m_power;
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

	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

	private void editedForward(Instances instances) throws IOException {
		BufferedReader datafile = readDataFile(M_EMPTY_REL);
		m_trainingInstances = new Instances(datafile);
		m_trainingInstances.setClassIndex(instances.classIndex());
		m_trainingInstances.add(instances.instance(0));
		for (int i = 1; i < instances.numInstances(); i++) {
			if (classify(instances.instance(i)) != instances.instance(i)
					.classValue()) {
				m_trainingInstances.add(instances.instance(i));
			}
		}
	}

	private void editedBackward(Instances instances) {
		m_trainingInstances = new Instances(instances);
		int i = 0;
		int numInstances = instances.numInstances();
		while (i < numInstances && numInstances > 1) {
			System.out.println(m_trainingInstances.instance(i));
			m_trainingInstances.delete(i);
			System.out.println(m_trainingInstances.instance(i));
			numInstances--;
			if (classify(instances.instance(i)) != instances.instance(i)
					.classValue()) {
				m_trainingInstances.add(instances.instance(i));
				i++;
				numInstances++;
			}
		}
	}

	public double crossValidationError(Instances instances, int numFolds)
			throws Exception {
		Random random = new Random();
		instances.randomize(random);
		double numInstancesPerFold = ((double) instances.numInstances())
				/ numFolds;
		BufferedReader datafile = readDataFile(M_EMPTY_REL);
		Instances emptyData = new Instances(datafile);
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
			buildClassifier(training);
			long curTime = System.nanoTime();
			errorSum += calcAvgError(test);
			long estTime = System.nanoTime() - curTime;
			sumTime += estTime;
			training.delete();
			test.delete();
		}
		long estTime = sumTime / numFolds;
		System.out.println("took " + estTime + " nano seconds");
		double crossValidError = ((double) errorSum) / numFolds;
		return crossValidError;
	}

	public int getM_k() {
		return m_k;
	}

	public void setM_k(int m_k) {
		this.m_k = m_k;
	}

	private void loadTestAndTraining(int numFolds, Instances[] folds,
			Instances training, Instances test, int i) {
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

	private void loadFolds(Instances instances, int numFolds,
			Instances emptyData, int curCutOff, Instances[] folds) {
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

	public double classify(Instance instance) {
		Instance[] kNearestNeighbors = new Instance[m_k];
		double[] kNearestNeighborsDistance = new double[m_k];
		findNearestNeighbors(instance, kNearestNeighbors,
				kNearestNeighborsDistance);

		int maxIndex = getWeightedClassVoteResult(instance, kNearestNeighbors,
				kNearestNeighborsDistance);

		double prediction = (double) maxIndex;

		return prediction;
	}

	private void findNearestNeighbors(Instance instance,
			Instance[] kNearestNeighbors, double[] kNearestNeighborsDistance) {
		int curIndex = 0;
		// assuming m_k < trainingInstance.numInstances()
		for (int i = 0; i < m_trainingInstances.numInstances(); i++) {
			if (i < m_k) {
				kNearestNeighbors[curIndex] = m_trainingInstances.instance(i);
				kNearestNeighborsDistance[curIndex] = distance(instance,
						m_trainingInstances.instance(i));
				curIndex++;

			} else {
				quickSort(kNearestNeighborsDistance, kNearestNeighbors);
				double curDistance = distance(instance,
						m_trainingInstances.instance(i));
				if (curDistance < kNearestNeighborsDistance[m_k - 1]) {
					kNearestNeighborsDistance[m_k - 1] = curDistance;
					kNearestNeighbors[m_k - 1] = m_trainingInstances
							.instance(i);
				}
			}
		}
	}

	private int getClassVoteResult(Instance instance,
			Instance[] kNearestNeighbors) {
		int numClassValues = instance.classAttribute().numValues();
		int[] numVotes = new int[numClassValues];
		for (int j = 0; j < m_k; j++) {
			// assuming class value can be cast to int
			if (kNearestNeighbors[j] != null) {
				numVotes[(int) kNearestNeighbors[j].classValue()]++;
			}

		}
		int max = numVotes[0];
		int maxIndex = 0;

		for (int j = 1; j < numClassValues; j++) {
			if (numVotes[j] > max) {
				max = numVotes[j];
				maxIndex = j;
			}
		}

		return maxIndex;
	}

	private int getWeightedClassVoteResult(Instance instance,
			Instance[] kNearestNeighbors, double[] kNearestNeighborsDistance) {
		int numClassValues = instance.classAttribute().numValues();
		double[] numVotes = new double[numClassValues];
		for (int j = 0; j < m_k; j++) {
			// assuming class value can be cast to int
			if (kNearestNeighbors[j] != null) {
				numVotes[(int) kNearestNeighbors[j].classValue()] += 1.0 / Math
						.pow(kNearestNeighborsDistance[j], 2);
			}

		}
		double max = numVotes[0];
		int maxIndex = 0;

		for (int j = 1; j < numClassValues; j++) {
			if (numVotes[j] > max) {
				max = numVotes[j];
				maxIndex = j;
			}
		}

		return maxIndex;
	}

	public void quickSort(double[] arrayToSort, Instance[] arrayToOrder) {
		int pivotIndex = arrayToSort.length - 1;
		if (arrayToSort.length > 1) {
			int oldPivot = pivotIndex;
			pivotIndex = reorderArray(arrayToSort, arrayToOrder, pivotIndex);
			if (oldPivot == pivotIndex) {
				pivotIndex--;
			}
			double[] leftPartToSort = new double[0];
			Instance[] leftPartToOrder = new Instance[0];
			if (pivotIndex > 0) {
				leftPartToSort = Arrays.copyOfRange(arrayToSort, 0, pivotIndex);
				leftPartToOrder = Arrays.copyOfRange(arrayToOrder, 0,
						pivotIndex);
				quickSort(leftPartToSort, leftPartToOrder);
			}

			double[] rightPartToSort = Arrays.copyOfRange(arrayToSort,
					pivotIndex + 1, arrayToSort.length);
			Instance[] rightPartToOrder = Arrays.copyOfRange(arrayToOrder,
					pivotIndex + 1, arrayToSort.length);
			quickSort(rightPartToSort, rightPartToOrder);

			if (pivotIndex > 0) {
				for (int j = 0; j < pivotIndex; j++) {
					arrayToSort[j] = leftPartToSort[j];
					arrayToOrder[j] = leftPartToOrder[j];
				}
			}

			int i = 0;
			for (int j = pivotIndex + 1; j < arrayToSort.length; j++) {
				arrayToSort[j] = rightPartToSort[i];
				arrayToOrder[j] = rightPartToOrder[i];
				i++;
			}

		}

	}

	private int reorderArray(double[] arrayToSort, Instance[] arrayToOrder,
			int pivotIndex) {
		int curIndex = 0;
		while (curIndex < pivotIndex) {
			if (arrayToSort[curIndex] > arrayToSort[pivotIndex]) {
				double beforePivot = arrayToSort[pivotIndex - 1];
				double bigger = arrayToSort[curIndex];
				double pivotValue = arrayToSort[pivotIndex];

				Instance beforePivotI = arrayToOrder[pivotIndex - 1];
				Instance biggerI = arrayToOrder[curIndex];
				Instance pivotValueI = arrayToOrder[pivotIndex];

				pivotIndex = pivotIndex - 1;
				arrayToSort[pivotIndex + 1] = bigger;
				arrayToSort[curIndex] = beforePivot;
				arrayToSort[pivotIndex] = pivotValue;

				arrayToOrder[pivotIndex + 1] = biggerI;
				arrayToOrder[curIndex] = beforePivotI;
				arrayToOrder[pivotIndex] = pivotValueI;

			} else {
				curIndex++;
			}
		}

		return pivotIndex;
	}

	private double distance(Instance instance1, Instance instance2) {
		double distance;
		switch (m_distanceType) {
		case "finite":
			distance = lNDistance(instance1, instance2);
			break;
		case "infinite":
			distance = lInfinityDistance(instance1, instance2);
			break;
		default:
			distance = lNDistance(instance1, instance2);
			break;
		}

		return distance;
	}

	private double lNDistance(Instance instance1, Instance instance2) {
		double sum = 0;
		for (int i = 0; i < instance1.numAttributes() - 1; i++) {
			double dif = (instance1.value(i) - instance2.value(i));
			dif = Math.abs(dif);
			sum += Math.pow(dif, m_power);

		}
		double rootPower = 1.0 / m_power;
		double distance = Math.pow(sum, rootPower);
		return distance;

	}

	private double lInfinityDistance(Instance instance1, Instance instance2) {
		double max = 0;
		for (int i = 0; i < instance1.numAttributes() - 1; i++) {
			double dif = (instance1.value(i) - instance2.value(i));
			dif = Math.abs(dif);
			if (dif > max) {
				max = dif;
			}
		}
		return max;
	}

	public double calcAvgError(Instances instances) {
		double sumError = 0;

		for (int i = 0; i < instances.numInstances(); i++) {
			Instance curInstance = instances.instance(i);
			boolean isDifferent = classify(curInstance) == curInstance
					.classValue() ? false : true;
			if (isDifferent) {
				sumError += 1;
			}
		}

		double avgError = sumError / instances.numInstances();
		return avgError;
	}

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

}
