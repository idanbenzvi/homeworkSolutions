package homework1;

import java.util.concurrent.ThreadLocalRandom;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression extends Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha = 0.0000001;
	private double m_epsilon = 0.03;

	public double getAlpha() {
		return m_alpha;
	}

	public void setAlpha(double alpha) {
		this.m_alpha = alpha;
	}

	private double m_error_coeff;
	private int m_num_iter;

	private boolean[] m_SelectedAttributes;
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public LinearRegression(double error_coeff, int num_iter) {
		m_error_coeff = error_coeff;
		m_num_iter = num_iter;
	}

	public LinearRegression() {
		m_error_coeff = 0.1;
		m_num_iter = 100;
	}

	// the method which runs to train the linear regression predictor, i.e.
	// finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		// since class attribute is also an attribuite we subtract 1
		m_truNumAttributes = trainingData.numAttributes() - 1;
		//findBestAlpha(trainingData);
		gradientDescent(trainingData);

	}

	public void findBestAlpha(Instances data) throws Exception {
		int expMin = -20;
		int expMax = 0;
		int j = expMin;
		double error = 1000;
		double tempError = 0;
		int curMin = expMin;
		while (j <= expMax) {

			m_alpha = Math.pow(10, j);
			// gradientDescentSetIter(data);
			gradientDescent(data);
			tempError = calculateSE(data);
			System.out.println(tempError);
			if (Double.isNaN(tempError)) {
				tempError = Double.MAX_VALUE;
			}
			if (tempError < error) {
				error = tempError;
				curMin = j;
			}
			j++;

		}

		m_alpha = Math.pow(10, curMin);

	}

	/**
	 * An implementation of the gradient descent algorithm which should try to
	 * return the weights of a linear regression predictor which minimizes the
	 * average squared error.
	 * 
	 * @param trainingData
	 * @return
	 * @throws Exception
	 */
	public void gradientDescent(Instances trainingData) throws Exception {

		int numInstances = trainingData.numInstances();
		double divisor = ((double) 1 / numInstances);

		m_coefficients = new double[m_truNumAttributes + 1];
		double[] gradients = new double[m_truNumAttributes + 1];
		double difError = 1000;
		while (difError > m_epsilon) {

			difError = gradDescentIteration(trainingData, divisor, gradients);

		}

	}

	public void gradientDescentSetIter(Instances trainingData) throws Exception {
		int numIter = 150000;
		int numInstances = trainingData.numInstances();
		double divisor = ((double) 1 / numInstances);

		m_coefficients = new double[m_truNumAttributes + 1];
		double[] gradients = new double[m_truNumAttributes + 1];
		double difError = 1000;
		for (int i = 0; i < numIter; i++) {

			difError = gradDescentIteration(trainingData, divisor, gradients);

		}

	}

	private double gradDescentIteration(Instances trainingData, double divisor, double[] gradients) throws Exception {
		double difError;
		for (int j = 0; j < m_truNumAttributes; j++) {
			double sumOfDifs = 0;
			for (int k = 0; k < trainingData.numInstances(); k++) {
				// calculating gradient for w_j
				Instance curInstance = trainingData.instance(k);
				double innerProd = innerProduct(m_coefficients, curInstance);

				double classValue = curInstance.classValue();
				double curValue = curInstance.value(j);
				double dif = (innerProd - classValue) * curValue;

				sumOfDifs += dif;
			}

			gradients[j] = divisor * sumOfDifs;

		}
		double sumOfDifs = 0;
		// calculating gradient for w_0
		for (int k = 0; k < trainingData.numInstances(); k++) {
			Instance curInstance = trainingData.instance(k);
			double innerProd = innerProduct(m_coefficients, curInstance);

			double dif = (innerProd - trainingData.instance(k).classValue());

			sumOfDifs += dif;
		}
		gradients[m_truNumAttributes] = divisor * sumOfDifs;
		double oldSsquaredError = calculateSE(trainingData);
		double[] tempWeights = new double[m_truNumAttributes + 1];
		// updating weights
		for (int j = 0; j < m_truNumAttributes + 1; j++) {
			tempWeights[j] = m_coefficients[j] - m_alpha * gradients[j];

		}
		for (int j = 0; j < m_truNumAttributes + 1; j++) {
			m_coefficients[j] = tempWeights[j];

		}
		double newSquaredError = calculateSE(trainingData);
		difError = oldSsquaredError - newSquaredError;

		return difError;
	}

	private double innerProduct(double[] coefficients, Instance instance) {
		double result = 0;
		for (int j = 0; j < instance.numAttributes() - 1; j++) {
			result += coefficients[j] * instance.value(j);
		}
		result += coefficients[instance.numAttributes() - 1];

		return result;
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by coefficients on a single instance.
	 * 
	 * @param instance
	 * @param coefficients
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double result = innerProduct(m_coefficients, instance);

		return result;
	}

	/**
	 * Calculates the total squared error over the test data on a linear
	 * regression predictor with weights given by coefficients.
	 * 
	 * @param testData
	 * @param coefficients
	 * @return
	 * @throws Exception
	 */
	public double calculateSE(Instances data) throws Exception {

		double mse = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			double prediction = regressionPrediction(data.instance(i));
			double error = prediction - data.instance(i).classValue();
			mse += Math.pow(error, 2);
		}
		double se = mse / data.numInstances();
		return se;
	}

	/**
	 * Finds the closed form solution to linear regression with one variable.
	 * Should return the coefficient that is to be multiplied by the input
	 * attribute.
	 * 
	 * @param data
	 * @return
	 */
	public double findClosedForm1D(Instances data) {
		double xSum = 0;
		double xYSum = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			xSum += data.instance(i).value(0) * data.instance(i).value(0);
			xYSum += data.instance(i).value(0) * data.instance(i).value(1);
		}
		double result = (1 / xSum) * xYSum;
		m_coefficients = new double[2];
		m_coefficients[0] = result;
		m_coefficients[1] = 0;
		m_ClassIndex = data.classIndex();
		// since class attribute is also an attribuite we subtract 1
		m_truNumAttributes = data.numAttributes() - 1;
		return result;
	}

}
