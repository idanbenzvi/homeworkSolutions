package hw2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

//import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class DecisionTree implements Classifier {

	Node m_decisionTree;
	int numAttributes = 0;

	double[] m_thresholdOfNumerics;
	List<Double>[] m_possibleThresholds;
	double[] m_averageOfNumerics;
	boolean m_isPruningOn = true;

	private class Node {
		int attributeIndex = -1;
		boolean isLeaf = false;
		boolean isRoot = false;
		Node parent;
		Double returnValue = null;
		Node[] children;
		int[][][] m_attrCount;
		int[][][] m_attrCountP;
		int[][][] m_attrCountE;
		int[] m_classCountE;

	}
	
	public void setPruningOn(boolean isOn){
		m_isPruningOn = isOn;
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

	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_thresholdOfNumerics = new double[trainingData.numAttributes()];
		
		Node node = new Node();
		node.isRoot = true;
		numAttributes = trainingData.numAttributes() - 1;
		m_possibleThresholds = new List[numAttributes];
		Instances[] dataSplit = new Instances[1];
		dataSplit[0] = new Instances(trainingData);
		measureCountsForSplits(trainingData, node, dataSplit);
		m_decisionTree = buildTree(trainingData, node, 0);
	}

	private Node buildTree(Instances data, Node parent, int splitIndex) throws Exception {
		
		
		double classValue = data.instance(0).classValue();
		boolean allClassSame = true;
		for (int i = 1; i < data.numInstances(); i++) {
			if (classValue != data.instance(i).classValue()) {
				allClassSame = false;
				break;
			}
		}
		if (allClassSame) {
			Node leafNode = new Node();
			leafNode.isLeaf = true;
			leafNode.returnValue = classValue;
			return leafNode;
		}

		boolean allAttrSame = true;
		attempt: {
			for (int j = 1; j < data.numInstances(); j++) {
				for (int i = 0; i < numAttributes; i++) {
					if (data.instance(0).value(i) != data.instance(j).value(i)) {
						allAttrSame = false;
						break attempt;
					}
				}
			}
		}
		if (allAttrSame) {
			Node leafNode = new Node();
			leafNode.isLeaf = true;
			
			leafNode.returnValue = classValue;
			return leafNode;
		}
		int bestAttrIndex = findBestAtr(parent.m_attrCount[splitIndex], parent.m_attrCountE[splitIndex],
				parent.m_attrCountP[splitIndex], parent.m_classCountE[splitIndex], data);
		Node node = new Node();
		node.attributeIndex = bestAttrIndex;
		node.isLeaf = false;
		node.children = new Node[data.attribute(bestAttrIndex).numValues()];
		node.parent = parent;

		Instances[] dataSplit = new Instances[node.children.length];
		// instantiate new instance of Instances
		for (int i = 0; i < dataSplit.length; i++) {
			BufferedReader datafile = readDataFile("empty_cancer.txt");

			Instances relData = new Instances(datafile);

			dataSplit[i] = new Instances(relData);
			dataSplit[i].setClassIndex(dataSplit[i].numAttributes() - 1);
		}
		measureCountsForSplits(data, node, dataSplit);
		int asdk = node.children.length;
		for (int k = 0; k < node.children.length; k++) {
			if (dataSplit[k].numInstances() > 0) {
				node.children[k] = buildTree(dataSplit[k], node, k);
				if (m_isPruningOn){
					pruneIfIrrelavant(node.children[k], dataSplit[k]);
				}
				

			} else {
				node.children[k] = new Node();
				int classCountE = 0;
				for (int i = 0; i < node.children.length; i++) {
					classCountE += node.m_classCountE[i];
				}
				int classCountP = data.numInstances() - classCountE;
				if (classCountE > classCountP) {
					// E has value 0
					node.children[k].returnValue = (double) 0;
				} else {
					// P has value 1
					node.children[k].returnValue = (double) 1;
				}
				node.children[k].isLeaf = true;
			}
		}

		// pruneIfIrrelavant(node, data);

		return node;
	}

	private void pruneIfIrrelavant(Node node, Instances data) throws Exception {
		if (node.isLeaf) {
			return;
		}
		
		double toBeat = 11.591;
		
		double myChiSquare = chiSquareTest(node, data);
		if (myChiSquare < toBeat) {
			node.isLeaf = true;
			int classCountE = 0;

			for (int i = 0; i < node.children.length; i++) {
				classCountE += node.m_classCountE[i];
			}
			int classCountP = data.numInstances() - classCountE;
			if (classCountE > classCountP) {
				// E has value 0
				node.returnValue = (double) 0;
			} else {
				// P has value 1
				node.returnValue = (double) 1;
			}

		}
		// node and make a new 'null' node which returns the majority vote
	}

	private void measureCountsForSplits(Instances data, Node node, Instances[] dataSplit) {
		int numDataSplits = 1;
		if (node.children != null) {
			numDataSplits = node.children.length;
		}
		int[][][] attrCount = new int[numDataSplits][numAttributes][];
		int[][][] attrCountP = new int[numDataSplits][numAttributes][];
		int[][][] attrCountE = new int[numDataSplits][numAttributes][];
		int[] classCountE = new int[numDataSplits];
		for (int i = 0; i < numAttributes; i++) {
			int numValues = 0;
			if (data.attribute(i).isNumeric()) {
				numValues = 2;
			} else {
				numValues = data.attribute(i).numValues();
			}
			for (int j = 0; j < numDataSplits; j++) {
				attrCount[j][i] = new int[numValues];
				attrCountP[j][i] = new int[numValues];
				attrCountE[j][i] = new int[numValues];
			}
		}

		for (int l = 0; l < data.numInstances(); l++) {
			int indexOfSplit = 0;
			if (node.attributeIndex != -1) {
				if (!data.attribute(node.attributeIndex).isNumeric()) {
					indexOfSplit = (int) data.instance(l).value(node.attributeIndex);
				} else {
					indexOfSplit = data.instance(l)
							.value(node.attributeIndex) < m_thresholdOfNumerics[node.attributeIndex] ? 0 : 1;
				}
				dataSplit[indexOfSplit].add(data.instance(l));
			}

			if (data.instance(l).classValue() == 0) {
				classCountE[indexOfSplit] += 1;
			}
			for (int i = 0; i < numAttributes; i++) {
				int attrValue = 0;
				if (data.attribute(i).isNumeric()) {
					attrValue = data.instance(l).value(i) < m_thresholdOfNumerics[i] ? 0 : 1;
				} else {
					attrValue = (int) data.instance(l).value(i);

				}
				attrCount[indexOfSplit][i][attrValue] += 1;
				if (data.instance(l).classValue() == 0) {
					attrCountE[indexOfSplit][i][attrValue] += 1;
				} else {
					attrCountP[indexOfSplit][i][attrValue] += 1;
				}
			}

		}

		node.m_attrCount = attrCount;
		node.m_attrCountE = attrCountE;
		node.m_attrCountP = attrCountP;
		node.m_classCountE = classCountE;
	}

	private void findPossibleThresholds(Instances data, int attrIndex) {
		Instances tempData = new Instances(data);
		tempData.sort(attrIndex);
		double classValue = tempData.instance(0).classValue();
		for (int i = 1; i < tempData.numInstances(); i++) {
			if (classValue != tempData.instance(i).classValue()) {
				m_possibleThresholds[attrIndex].add(tempData.instance(i).value(attrIndex));
			}
		}
	}

	private int findBestAtr(int[][] attrCount, int[][] attrCountE, int[][] attrCountP, int classCountE,
			Instances data) {
		double maxInfoGain = 0;
		int maxInfoIndex = 0;
		double[] tmpThresholds = new double[numAttributes];
		double[] bestInfoGainNumerics = new double[numAttributes];
		for (int j = 0; j < numAttributes; j++) {
			double curBestThresholdValue = 0;
			double tmpGainOfThresh = 0;

			if (data.attribute(j).isNumeric()) {
				m_possibleThresholds[j] = new ArrayList<Double>(100);
				findPossibleThresholds(data, j);

				double tmpMaxGain = 0;
				for (Double thresholdValue : m_possibleThresholds[j]) {

					int[] tmpAtrCount = new int[2];
					int[] tmpAtrCountE = new int[2];
					int[] tmpAtrCountP = new int[2];
					int tmpClasSCountE = 0;
					measureCountOfOneAttribute(tmpAtrCount, tmpAtrCountE, tmpAtrCountP, tmpClasSCountE, data, j,
							thresholdValue);
					tmpGainOfThresh = calcInfoGain(tmpAtrCount, tmpAtrCountE, tmpAtrCountP, classCountE, j, data);
					if (tmpGainOfThresh > tmpMaxGain) {
						tmpMaxGain = tmpGainOfThresh;
						curBestThresholdValue = thresholdValue;
					}
				}
				bestInfoGainNumerics[j] = tmpMaxGain;
				tmpThresholds[j] = curBestThresholdValue;
			}
		}
		for (int j = 0; j < numAttributes; j++) {
			double tmpInfoGain = 0;
			if (data.attribute(j).isNumeric()) {
				tmpInfoGain = bestInfoGainNumerics[j];
			} else {
				tmpInfoGain = calcInfoGain(attrCount[j], attrCountE[j], attrCountP[j], classCountE, j, data);
			}
			
			if (tmpInfoGain > maxInfoGain) {
				maxInfoIndex = j;
				maxInfoGain = tmpInfoGain;
			}
		}
		if (data.attribute(maxInfoIndex).isNumeric()) {
			m_thresholdOfNumerics[maxInfoIndex] = tmpThresholds[maxInfoIndex];
		}
		
		return maxInfoIndex;
	}

	private void measureCountOfOneAttribute(int[] attrCount, int[] attrCountE, int[] attrCountP, int classCountE,
			Instances data, int attrIndex, double thresholdValue) {
		for (int i = 0; i < data.numInstances(); i++) {
			Instance curInstance = data.instance(i);
			if (curInstance.value(attrIndex) < thresholdValue) {
				attrCount[0] += 1;
				if (curInstance.classValue() == 0) {
					attrCountE[0] += 1;
					classCountE += 1;
				} else {
					attrCountP[0] += 1;
				}
			} else {
				attrCount[1] += 1;
				if (curInstance.classValue() == 0) {
					attrCountE[1] += 1;
					classCountE += 1;
				} else {
					attrCountP[1] += 1;
				}
			}
		}
	}

	private double calcInfoGain(int[] attrCount, int[] attrCountE, int[] attrCountP, int classCountE, int attrIndex,
			Instances data) {
		Attribute attribute = data.attribute(attrIndex);
		int sizeOfAttr;
		if (attribute.isNumeric()) {
			sizeOfAttr = 2;
		} else {
			sizeOfAttr = attribute.numValues();
		}
		double probE = ((double) classCountE) / data.numInstances();
		double probP = ((double) data.numInstances() - classCountE) / data.numInstances();
		double entropyClass = calcEntropy(probE, probP);
		double infoGain = entropyClass;
		for (int i = 0; i < sizeOfAttr; i++) {

			double probOfAttrValI = ((double) attrCount[i]) / data.numInstances();
			double divisor = 1;

			if (attrCount[i] != 0) {
				divisor = attrCount[i];
			}
			double condProbOfE = ((double) attrCountE[i]) / divisor;
			double condProbOfP = ((double) attrCountP[i]) / divisor;
			infoGain += -1 * probOfAttrValI * calcEntropy(condProbOfE, condProbOfP);
		}
		return infoGain;
	}

	private double calcEntropy(double prob1, double prob2) {
		double entropy = 0;
		if (prob1 == 0) {
			prob1 = 1;
		}
		if (prob2 == 0) {
			prob2 = 1;
		}
		entropy += -1 * prob1 * (Math.log(prob1) / Math.log(2));
		entropy += -1 * prob2 * (Math.log(prob2) / Math.log(2));
		return entropy;
	}

	public double classify(Instance instance) {

		Node curNode = m_decisionTree;
		boolean isLeaf = curNode.isLeaf;
		int attrVal = -1;
		while (!isLeaf) {
			attrVal = (int) instance.value(curNode.attributeIndex);
			Node temp = curNode.children[attrVal];
			
			curNode = curNode.children[attrVal];
			isLeaf = curNode.isLeaf;
		}

		return curNode.returnValue;
	}

	private double chiSquareTest(Node node, Instances data) {
		double chiSq = 0;
		double probOfE;
		double probOfP;
		double expecE;
		double expecP;
		int countE;
		int countP;
		int attrCount;
		int classCountE = 0;
		
		int numInstances = data.numInstances();
		for (int i = 0; i < node.children.length; i++) {
			classCountE += node.m_classCountE[i];
		}
		// prob of e
		double probE = ((double) classCountE) / numInstances;
		// prob of p
		double probP = ((double) numInstances - classCountE) / numInstances;
		for (int i = 0; i < node.m_attrCount.length; i++) {
			// pi
			countE = node.m_attrCountE[i][node.attributeIndex][i];
			// ni
			countP = node.m_attrCountP[i][node.attributeIndex][i];
			// si
			attrCount = node.m_attrCount[i][node.attributeIndex][i];

			expecE = probE * attrCount;
			expecP = probP * attrCount;
			if (expecE != 0) {
				chiSq += (Math.pow((countE - expecE), 2)) / expecE;
			}
			if (expecP != 0) {
				chiSq += (Math.pow((countP - expecP), 2)) / expecP;
			}
		}
		return chiSq;
	}
	
	public double calcAverageError(Instances instances){
		int numErrors = 0;
		
		for (int i = 0; i < instances.numInstances(); i++) {
			double prediction = classify(instances.instance(i));
			double classValue = instances.instance(i).classValue();
			if (prediction != classValue) {
				numErrors += 1;
			}
		}
		
		return ((double) numErrors) / instances.numInstances();
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
