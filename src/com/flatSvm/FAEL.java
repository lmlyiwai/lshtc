package com.flatSvm;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.tools.Matrix;
import com.tools.ProcessProblem;
import com.tools.RandomSequence;
import com.tools.Sigmoid;


/**
 * Hierarchical Multi-label Classification using Fully Associative Ensemble Learning 
 */
public class FAEL {
	private Problem 		prob;
	private Parameter 		param;
	private int[]			ulabels;
	private double[][]		w;
	
	public FAEL(Problem prob, Parameter param) {
		this.prob = prob;
		this.param = param;
		this.ulabels = ProcessProblem.getUniqueLabels(prob.y);
	}
	
	/**
	 * 
	 */
	public double[] corssValidation(Problem prob, Parameter param, int n_fold, double c , double g, double lam) {
		
		param.setC(c);
		int n = prob.l;
		int[][] pre = new int[n][];
		
		int[] index = RandomSequence.randomSequence(n);
		
		int segLength = n / n_fold;
		
		int vbegin = 0;
		int vend = 0;		
		
		int[] validIndex = null;
		int[] trainIndex = null;
		int counter = 0;
		for(int i = 0; i < n_fold; i++) {
			vbegin = i * segLength;
			vend = i * segLength + segLength;
			
			validIndex = new int[vend - vbegin];
			trainIndex = new int[n - validIndex.length];
			
			counter = 0;
			for(int j = vbegin; j < vend; j++) {
				validIndex[counter++] = index[j];
			}
			
			counter = 0;
			for(int j = 0; j < vbegin; j++) {
				trainIndex[counter++] = index[j];
			}
			for(int j = vend; j < n; j++) {
				trainIndex[counter++] = index[j];
			}
			
			Problem train = new Problem();
			train.l = trainIndex.length;
			train.n = prob.n;
			train.bias = prob.bias;
			train.x = new DataPoint[trainIndex.length][];
			train.y = new int[trainIndex.length][];
			
			counter = 0;
			for(int j = 0; j < trainIndex.length; j++) {
				train.x[counter] = prob.x[trainIndex[j]];
				train.y[counter] = prob.y[trainIndex[j]];
				counter++;
			}
			
			Problem valid = new Problem();
			valid.l = validIndex.length;
			valid.n = prob.n;
			valid.bias = prob.bias;
			valid.x = new DataPoint[validIndex.length][];
			valid.y = new int[validIndex.length][];
			
			counter = 0;
			for(int j = 0; j < validIndex.length; j++) {
				valid.x[counter] = prob.x[validIndex[j]];
				valid.y[counter] = prob.y[validIndex[j]];
				counter++;
			}
			
			FlatSVM fs = new FlatSVM(prob, param);
			
			DataPoint[][] w = fs.train(train, param);
			
			double[][] Z = fs.predictValues(train.x);
			Sigmoid.sigmoid(Z, g);
			
			double[][] Y = ProcessProblem.labelToMatrix(train.y, this.ulabels);
			double[][] I = Matrix.identityMatrix(this.ulabels.length);
			Matrix.scaleMatrix(I, lam);
			
			double[][] item1 = Matrix.multi(Matrix.trans(Z), Z);
			item1 = Matrix.matrixAdd(item1, I);
			item1 = Matrix.inv(item1);
			
			double[][] wMatrix = Matrix.multi(Matrix.multi(item1, Matrix.trans(Z)), Y);
			
			double[][] trainPreVal = Matrix.multi(Z, wMatrix);
			double[] th = getThresholds(trainPreVal, Y);
			
			double[][] validZ = fs.predictValues(valid.x);
			Sigmoid.sigmoid(validZ, g);
			
			double[][] validPreVal = Matrix.multi(validZ, wMatrix);
			
			int[][] predictLabel = predict(validPreVal, 0.5);
//			int[][] predictLabel = predict(validPreVal, th);
			
			for(int j = 0; j < validIndex.length; j++) {
				pre[validIndex[j]] = predictLabel[j];
			}
		}
		double microf1 = Measures.microf1(this.ulabels, prob.y, pre);
		double macrof1 = Measures.macrof1(this.ulabels, prob.y, pre);
		double hammingloss = Measures.averageSymLoss(prob.y, pre);
		System.out.println("c = " + param.getC() + ", g = " + g +
				", lam = " + lam + ", Micro-F1 = " + microf1 + ", Macro-F1 = " 
				+ macrof1 + ", Hamming Loss = " + hammingloss);
		
		double[] perf = {microf1, macrof1, hammingloss};
		return perf;
	}
	
	/**
	 * 
	 */
	public int[][] predict(double[][] pv, double threshold) {
		int[][] pl = new int[pv.length][];
		for(int i = 0; i < pl.length; i++) {
			pl[i] = predictSingleSample(pv[i], threshold);
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public int[][] predict(double[][] pv, double[] threshold) {
		int[][] pl = new int[pv.length][];
		for(int i = 0; i < pl.length; i++) {
			pl[i] = predictSingleSample(pv[i], threshold);
		}
		return pl;
	}
	
	/**
	 * 
	 */
	public int[] predictSingleSample(double[] p, double[] threshold) {
		int counter = 0;
		for(int i = 0; i < p.length; i++) {
			if(p[i] > threshold[i]) {
				counter = counter + 1;
			}
		}
		
		int[] result = new int[counter];
		counter = 0;
		for(int i = 0; i < p.length; i++) {
			if(p[i] > threshold[i]) {
				result[counter] = this.ulabels[i];
				counter = counter + 1;
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public int[] predictSingleSample(double[] p, double threshold) {
		int counter = 0;
		for(int i = 0; i < p.length; i++) {
			if(p[i] > threshold) {
				counter = counter + 1;
			}
		}
		
		int[] result = new int[counter];
		counter = 0;
		for(int i = 0; i < p.length; i++) {
			if(p[i] > threshold) {
				result[counter] = this.ulabels[i];
				counter = counter + 1;
			}
		}
		return result;
	}
	
	/**
	 * 
	 */
	public double[] getThresholds(double[][] pv, double[][] y) {
		double[] thresh = new double[this.ulabels.length];
		double posSum = 0;
		double posCou = 0;
		double negSum = 0;
		double negCou = 0;
		for(int i = 0; i < thresh.length; i++) {
			double[] tpv = Matrix.getMatrixColumn(pv, i);
			double[] ty = Matrix.getMatrixColumn(y, i);
			posSum = 0;
			negSum = 0;
			for(int j = 0; j < tpv.length; j++) {
				if(ty[j] == 1) {
					posCou = posCou + 1;
					posSum = posSum + tpv[j];
				} else {
					negCou = negCou + 1;
					negSum = negSum + tpv[j];
				}
			}
			thresh[i] = 0.5 * (posSum / posCou + negSum / negCou);
		}
		return thresh;
	}
}
