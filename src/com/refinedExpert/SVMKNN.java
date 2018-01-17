package com.refinedExpert;

import java.text.DecimalFormat;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

public class SVMKNN {
	private Problem train;
	private Parameter param;
	private int[] k;
	private int[] ulabels;
	private static DecimalFormat df = new DecimalFormat("0.0000");
	public SVMKNN(Problem train, Parameter param, int[] k) {
		this.train = train;
		this.param = param;
		this.k = k;
		this.ulabels = Tools.getUniqueItem(train.y);
	}
	
	/**
	 * 
	 */
	public double[][] train() {
		DataPoint[][] weight = Tools.train(this.train, this.param, this.ulabels);
		double[][] tfMat = Tools.transform(this.train.x, weight, Tools.NONORMALIE);
		double[][] perfs = Tools.predict(tfMat, this.train.y, tfMat, this.train.y, this.k, false, false);
		return perfs;
	}
	
	/**
	 * 
	 */
	public double[][] trainAndTest(Problem test) {
		DataPoint[][] weight = Tools.train(this.train, this.param, this.ulabels);
		double[][] trainMat = Tools.transform(this.train.x, weight, Tools.NONORMALIE);
		double[][] testMat = Tools.transform(test.x, weight, Tools.NONORMALIE);
		double[][] testPerformance = new double[this.k.length][];
		for (int i = 0; i < this.k.length; i++) {
			double[] trainPerfs = Tools.predict(trainMat, this.train.y, trainMat, this.train.y, k[i], false, false);
			double[] testPerfs = Tools.predict(trainMat, this.train.y, testMat, test.y, k[i], true, false);
			testPerformance[i] = testPerfs;
			System.out.println("c = " + this.param.getC() + ", k = " + k[i] 
					+ ", MiF1 = " + df.format(trainPerfs[0]) + ", MaF1 = " + df.format(trainPerfs[1])
					+", HamLoss = " + df.format(trainPerfs[2]) + ", 0/1 Loss = " + df.format(trainPerfs[3])
					+ ", mif1 = " + df.format(testPerfs[0]) + ", maf1 = " + df.format(testPerfs[1])
					+ ", hamloss = " + df.format(testPerfs[2]) + ", 0/1 loss = " + df.format(testPerfs[3]));
		}
		return testPerformance;
	}
	
	/**
	 * 
	 */
	public double[][] trainAndTestNormalize(Problem test) {
		DataPoint[][] weight = Tools.train(this.train, this.param, this.ulabels);
		double[][] trainMat = Tools.transform(this.train.x, weight, Tools.HORIZONTAL);
		double[][] testMat = Tools.transform(test.x, weight, Tools.HORIZONTAL);
		double[][] testPerformance = new double[this.k.length][];
		for (int i = 0; i < this.k.length; i++) {
			double[] trainPerfs = Tools.predict(trainMat, this.train.y, trainMat, this.train.y, k[i], false, false);
			double[] testPerfs = Tools.predict(trainMat, this.train.y, testMat, test.y, k[i], true, false);
			testPerformance[i] = testPerfs;
			System.out.println("c = " + this.param.getC() + ", k = " + k[i] 
					+ ", MiF1 = " + df.format(trainPerfs[0]) + ", MaF1 = " + df.format(trainPerfs[1])
					+", HamLoss = " + df.format(trainPerfs[2]) + ", 0/1 Loss = " + df.format(trainPerfs[3])
					+ ", mif1 = " + df.format(testPerfs[0]) + ", maf1 = " + df.format(testPerfs[1])
					+ ", hamloss = " + df.format(testPerfs[2]) + ", 0/1 loss = " + df.format(testPerfs[3]));
		}
		return testPerformance;
	}
}
