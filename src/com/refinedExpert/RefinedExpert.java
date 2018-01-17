/**
 * ���Դ���༶������ѵ�������Ͳ��������ֲ���һ������
 * 1������Ϊÿһ��ѵ��һ��֧����������Ϊ������
 * 2����ԭʼ������Ϊ֧�������������룬����������֮ǰ��֮ͬ����������������ĳ��֧����������ѵ����
 * 		 ���������֧���������������롣
 * 3��ѵ������������
 * �����һ����֧�����������ڶ�����KNN
 */
package com.refinedExpert;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.dmoz.TestPath;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.RandomSequence;
import com.tools.Sort;

public class RefinedExpert {
	public static final int MICROF1 = 0;
	public static final int MACROF1 = 1;
	public static final int HAMM = 2;
	
	private static DecimalFormat df = new DecimalFormat("0.0000");
	private Problem prob;
	private Parameter param;
	private double[] k;
	private int fold;
	private int[] uniqueLabels;
	private DataPoint[][] w;
	private double[][] trainpv;
	private int bestk;
	private int critical;
	private int[][] extendy;
	
	
	public RefinedExpert(Problem prob, Parameter param, double[] k, int fold, int critical) {
		this.prob = prob;
		this.param = param;
		this.k = k;
		this.fold = fold;
		this.uniqueLabels = getUniqueLabels();
		this.w = null;
		this.critical = critical;
	}
	
	/**
	 * 
	 */
	public void trainAndTest(Problem test) {
		if (this.prob == null || this.param == null || this.k == null) {
			return;
		}
		
		getWeight();
		double[][] ts = transformSamples();
		scale(ts);
		this.trainpv = ts;
		this.extendy = this.prob.y;
		double[][] performance = new double[this.k.length][];
		double[][] testperf = new double[this.k.length][];
		for (int i = 0; i < performance.length; i++) {
//System.out.println("Start training...");
long startTime = System.currentTimeMillis();
			performance[i] = leaveOneOut(ts, k[i]);
//System.out.println("Start predicting...");
			testperf[i] = predict(test, (int)k[i]);
long endTime = System.currentTimeMillis();
System.out.print("c = " + this.param.getC() + ", k = " + k[i] + ", MiF1 = " + df.format(performance[i][0]) + 
		", MaF1 = " + df.format(performance[i][1]) + ", HamLoss = " + df.format(performance[i][2]) +
		", 0/1 loss = " + df.format(performance[i][3]) + 
		", mif1 = " + df.format(testperf[i][0]) + ", maf1 = " + df.format(testperf[i][1]) + 
		", HamLoss = " + df.format(testperf[i][2]) +
		", 0/l loss = " + df.format(testperf[i][3]) + 
		"\n");
		}
	}
	
	/**
	 * �м����������˹����
	 */
	public void trainAndTestWithNoise(Problem test) {
		if (this.prob == null || this.param == null || this.k == null) {
			return;
		}
		
		getWeight();
		double[][] ts = transformSamples();
		scale(ts);
		this.trainpv = ts;
		this.extendy = this.prob.y;
		double[][] performance = new double[this.k.length][];
		double[][] testperf = new double[this.k.length][];
		for (int i = 0; i < performance.length; i++) {
//System.out.println("Start training...");
long startTime = System.currentTimeMillis();
			performance[i] = leaveOneOut(ts, k[i]);
//System.out.println("Start predicting...");
			testperf[i] = predict(test, (int)k[i]);
long endTime = System.currentTimeMillis();
System.out.print("c = " + this.param.getC() + ", k = " + k[i] + ", MiF1 = " + df.format(performance[i][0]) + 
		", MaF1 = " + df.format(performance[i][1]) + ", HamLoss = " + df.format(performance[i][2]) +
		", 0/1 loss = " + df.format(performance[i][3]) + 
		", mif1 = " + df.format(testperf[i][0]) + ", maf1 = " + df.format(testperf[i][1]) + 
		", HamLoss = " + df.format(testperf[i][2]) +
		", 0/l loss = " + df.format(testperf[i][3]) + 
		"\n");
		}
	}
	
	
	/**
	 * 
	 */
	public void trainAndTestSingleLabel(Problem test) {
		if (this.prob == null || this.param == null || this.k == null) {
			return;
		}
		
		getWeight();
		double[][] ts = transformSamples();
		this.trainpv = ts;
		this.extendy = this.prob.y;
		double[][] performance = new double[this.k.length][];
		double[][] testperf = new double[this.k.length][];
		for (int i = 0; i < performance.length; i++) {
System.out.println("Start training...");
long startTime = System.currentTimeMillis();
			performance[i] = leaveOneOut(ts, k[i]);
System.out.println("Start predicting...");
			testperf[i] = predictMax(test, (int)k[i]);
long endTime = System.currentTimeMillis();
System.out.print("c = " + this.param.getC() + ", k = " + k[i] + ", MiF1 = " + df.format(performance[i][0]) + 
		", MaF1 = " + df.format(performance[i][1]) + ", HamLoss = " + df.format(performance[i][2]) +
		", 0/1 loss = " + df.format(performance[i][3]) + 
		", mif1 = " + df.format(testperf[i][0]) + ", maf1 = " + df.format(testperf[i][1]) + 
		", HamLoss = " + df.format(testperf[i][2]) +
		", 0/l loss = " + df.format(testperf[i][3]) + 
		"\n");
		}
	}
	
	/**
	 * ��������������һ��
	 */
	public void trainAndTestExtend(Problem test) {
		if (this.prob == null || this.param == null || this.k == null) {
			return;
		}
		
		getWeight();
		double[][] ts = transformSamples();
		double[][] ets = transTestSamples(this.prob.x);
		double[][] extedMat = mergeMat(ets, ts);
		this.trainpv = extedMat;
		this.extendy = extendMat(this.prob.y);
		
		double[][] performance = new double[this.k.length][];
		double[][] testperf = new double[this.k.length][];
		for (int i = 0; i < performance.length; i++) {
long startTime = System.currentTimeMillis();
			performance[i] = leaveOneOut(this.trainpv, k[i]);
			testperf[i] = predict(test, (int)k[i]);
long endTime = System.currentTimeMillis();
System.out.print("c = " + this.param.getC() + ", k = " + k[i] + ", MiF1 = " + df.format(performance[i][0]) + 
		", MaF1 = " + df.format(performance[i][1]) + ", HamLoss = " + df.format(performance[i][2]) +
		", 0/1 loss = " + df.format(performance[i][3]) + 
		", mif1 = " + df.format(testperf[i][0]) + ", maf1 = " + df.format(testperf[i][1]) + 
		", HamLoss = " + df.format(testperf[i][2]) +
		", 0/l loss = " + df.format(testperf[i][3]) + 
		"\n");
		}

	}
	
	/**
	 * @return ����k*3�ľ������е��б�ʾ����kʱ����MiF1��MaF1, Hamming Loss
	 */
	public double[][] train() {
		if (this.prob == null || this.param == null || this.k == null) {
			return null;
		}
		
		double[][] ts = transformSamples();
		scale(ts);
		double[][] performance = new double[this.k.length][];
		this.trainpv = ts;
		this.extendy = this.prob.y;
		
		for (int i = 0; i < performance.length; i++) {
			performance[i] = leaveOneOut(ts, k[i]);
			System.out.println("c = " + this.param.getC() + ", k = " + k[i] 
					+ ", MiF1 = " + df.format(performance[i][0])
					+ ", MaF1 = " + df.format(performance[i][1]) 
					+ ",  HamLoss = " + df.format(performance[i][2])
					+ ", 0/1 loss = " + df.format(performance[i][3]));
		}
		return performance;
	}
	
	/**
	 * 
	 */
	private double[] leaveOneOut(double[][] ts, double k) {
		if (ts == null) {
			return null;
		}
		
		int[][] predictedLabel = new int[ts.length][];
		for (int i = 0; i < ts.length; i++) {
			double[] samplei = ts[i];
			double[] distance = new double[ts.length];
			for (int j = 0; j < ts.length; j++) {
				double[] sub = SparseVector.subVector(samplei, ts[j]);
				distance[j] = Math.sqrt(SparseVector.innerProduct(sub, sub));
			}
			int[] index = Sort.getIndexBeforeSort(distance);
			int[][] fkl = getFirstKlabel(this.extendy, index, (int)k, false);
			predictedLabel[i] = getLabel(fkl);                     //multi label
//			predictedLabel[i] = getLabelMax(fkl);              //single label
		}
		
		double microf1 = Measures.microf1(this.uniqueLabels, this.extendy, predictedLabel);
		double macrof1 = Measures.macrof1(this.uniqueLabels, this.extendy, predictedLabel);
		double hamloss = Measures.averageSymLoss(this.extendy, predictedLabel);
		double zoloss = Measures.zeroOneLoss(this.extendy, predictedLabel);
		double[] performance = {microf1, macrof1, hamloss, zoloss};
		return performance;
	}
	
	/**
	 * @param y��ά����
	 * @return ����
	 */
	private int[][] extendMat(int[][] y) {
		if (y == null) {
			return null;
		}
		int[][] twoy = new int[y.length * 2][];
		for (int i = 0; i < y.length; i++) {
			twoy[i] = new int[y[i].length];
			for (int j = 0; j < twoy[i].length; j++) {
				twoy[i][j] = y[i][j];
			}
		}
		for (int i  = y.length; i < twoy.length; i++) {
			twoy[i] = new int[y[i-y.length].length];
			for (int j = 0; j < twoy[i].length; j++) {
				twoy[i][j] = y[i-y.length][j];
			}
		}
		return twoy;
	}
	
	/**
	 * �ϲ�����matb��matb���������ã��������ݱ���
	 * @param
	 */
	private double[][] mergeMat(double[][] mata, double[][] matb) {
		if (mata == null || matb == null) {
			return null;
		}
		int lengtha = mata.length;
		int lengthb = matb.length;
		double[][] mergedAB = new double[lengtha + lengthb][];
		for (int i = 0; i < lengtha; i++) {
			mergedAB[i] = mata[i];
		}
		for (int i = lengtha; i < lengtha + lengthb; i++) {
			mergedAB[i] = matb[i-lengtha];
		}
		return mergedAB;
	}
	
	/**
	 * 
	 */
	public int[] getLabel(int[][] fkl) {
		if (fkl == null) {
			return null;
		}
		
		double threshold = ((double)fkl.length) / 2;
		Map<Integer, Double> countOfEachLabel = new HashMap<Integer, Double>();
		for (int i = 0; i < fkl.length; i++) {
			for (int j = 0; j < fkl[i].length; j++) {
				int key = fkl[i][j];
				if (countOfEachLabel.containsKey(key)) {
					double value = countOfEachLabel.get(key);
					value += 1;
					countOfEachLabel.put(key, value);
				} else {
					countOfEachLabel.put(key, 1.0);
				}
			}
		}
		
		List<Integer> finalLabel = new ArrayList<Integer>();
		Set<Integer> keySet = countOfEachLabel.keySet();
		Iterator<Integer> it = keySet.iterator();
		while (it.hasNext()) {
			int key = it.next();
			double value = countOfEachLabel.get(key);
			if (value > threshold) {
				finalLabel.add(key);
			}
		}
		
		int[] labelArray = new int[finalLabel.size()];
		for (int i = 0; i < labelArray.length; i++) {
			labelArray[i] = finalLabel.get(i);
		}
		return labelArray;
	}
	
	/**
	 * 
	 */
	public int[] getLabelMax(int[][] fkl) {
		if (fkl == null) {
			return null;
		}
		
		Map<Integer, Double> countOfEachLabel = new HashMap<Integer, Double>();
		for (int i = 0; i < fkl.length; i++) {
			for (int j = 0; j < fkl[i].length; j++) {
				int key = fkl[i][j];
				if (countOfEachLabel.containsKey(key)) {
					double value = countOfEachLabel.get(key);
					value += 1;
					countOfEachLabel.put(key, value);
				} else {
					countOfEachLabel.put(key, 1.0);
				}
			}
		}
		
		List<Integer> finalLabel = new ArrayList<Integer>();
		Set<Integer> keySet = countOfEachLabel.keySet();
		double max = Double.NEGATIVE_INFINITY;
		int label = -1;
		Iterator<Integer> it = keySet.iterator();
		while (it.hasNext()) {
			int key = it.next();
			double value = countOfEachLabel.get(key);
			if (value > max) {
				max = value;
				label = key;
			}
		}
		
		int[] labelArray = {label};
		return labelArray;
	}
	
	/**
	 * ��������K���������
	 */
	public int[][] getFirstKlabel(int[][] y, int[] index, int k, boolean startFromZero) {
		if (y == null || index == null) {
			return null;
		}
		
		int[][] fkl = new int[k][];
		if (startFromZero) {
			for (int i = 0; i < k; i++) {
				fkl[i] = y[index[i]];
			}
		} else {
			for (int i = 1; i <= k; i++) {
				fkl[i-1] = y[index[i]];
			}
		}
		return fkl;
	}
	
	/**
	 * @return ��һ����������
	 */
	public double[][] transformSamples() {
		int n = prob.l;
		int[] index = RandomSequence.randomSequence(n);
		int segLength = n / fold;
		int vbegin = 0;
		int vend = 0;		
		int[] validIndex = null;
		int[] trainIndex = null;
		int counter = 0;
		
		double[][] transfored = new double[n][];
		
		for(int i = 0; i < fold; i++) {
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
			
			Problem localTrain = new Problem();
			localTrain.l = trainIndex.length;
			localTrain.n = prob.n;
			localTrain.bias = prob.bias;
			localTrain.x = new DataPoint[trainIndex.length][];
			localTrain.y = new int[trainIndex.length][];
			
			counter = 0;
			for(int j = 0; j < trainIndex.length; j++) {
				localTrain.x[counter] = prob.x[trainIndex[j]];
				localTrain.y[counter] = prob.y[trainIndex[j]];
				counter++;
			}
			
			Problem localValid = new Problem();
			localValid.l = validIndex.length;
			localValid.n = prob.n;
			localValid.bias = prob.bias;
			localValid.x = new DataPoint[validIndex.length][];
			localValid.y = new int[validIndex.length][];
			
			counter = 0;
			for(int j = 0; j < validIndex.length; j++) {
				localValid.x[counter] = prob.x[validIndex[j]];
				localValid.y[counter] = prob.y[validIndex[j]];
				counter++;
			}
			
			double[][] validPredictValue = translate(localTrain, localValid);
			for (int j = 0; j < validPredictValue.length; j++) {
				transfored[validIndex[j]] = validPredictValue[j];
			}
		}
		return transfored;
	}
	
	/**
	 * @param trianѵ��������validȷ������
	 * @return valid����ת��֮��ľ���
	 */
	public double[][] translate(Problem train, Problem valid) {
		if (train == null || valid == null) {
			return null;
		}
		
		DataPoint[][] w = new DataPoint[this.uniqueLabels.length][];
		for (int i = 0; i < w.length; i++) {
			int label = this.uniqueLabels[i];
			int[] bl = getBinaryLabel(train.y, label);
			double[] loss = new double[1];
			w[i] = Linear.train(train, bl, param, null, loss, null, 0);
		}
		
		double[][] predictValue = new double[valid.l][w.length];
		for (int i = 0; i < predictValue.length; i++) {
			for (int j = 0; j < w.length; j++) {
				predictValue[i][j] = SparseVector.innerProduct(valid.x[i], w[j]);
			}
		}
		return predictValue;
	}
	
	/**
	 * @param y���, labelָ�����
	 * @return
	 */
	private int[] getBinaryLabel(int[][] y, int label) {
		if (y == null) {
			return null;
		}
		
		int[] bl = new int[y.length];
		for (int i = 0; i < bl.length; i++) {
			if (contain(y[i], label)) {
				bl[i] = 1;
			} else {
				bl[i] = -1;
			}
		}
		return bl;
	}
	
	/**
	 * 
	 */
	private boolean contain(int[] y, int label) {
		if (y == null) {
			return false;
		}
		
		boolean ifContain = false;
		for (int i = 0; i < y.length; i++) {
			if (y[i] == label) {
				ifContain = true;
				break;
			}
		}
		return ifContain;
	}
	
	/**
	 * @return ����ѵ�����������г��ֹ������
	 */
	private int[] getUniqueLabels() {
		if (prob == null || prob.y == null) {
			return null;
		}
		
		Set<Integer> labelSet = new HashSet<Integer>();
		for (int i = 0; i < this.prob.y.length; i++) {
			for (int j = 0; j < this.prob.y[i].length; j++) {
				labelSet.add(this.prob.y[i][j]);
			}
		}
		
		int[] label = new int[labelSet.size()];
		Iterator<Integer> it = labelSet.iterator();
		int index = 0;
		while (it.hasNext()) {
			label[index++] = it.next();
		}
		return label;
	}
	
	/**
	 * @param testSample��������������k������
	 * @return Ԥ�����
	 */
	public double[] predict(Problem test, int k) {
		if (test == null) {
			return null;
		}
		
		double[][] pv = transTestSamples(test.x);
		scale(pv);
		int[][] predictedLabel = new int[test.l][];
		for (int i = 0; i < predictedLabel.length; i++) {
			double[] samplei = pv[i];
			double[] distance = new double[this.trainpv.length];
			for (int j = 0; j < this.trainpv.length; j++) {
				double[] sub = SparseVector.subVector(samplei, this.trainpv[j]);
				distance[j] = Math.sqrt(SparseVector.innerProduct(sub, sub));
			}
			int[] index = Sort.getIndexBeforeSort(distance);
			int[][] fkl = getFirstKlabel(this.extendy, index, (int)k, true);
			predictedLabel[i] = getLabel(fkl);											//�����
//			predictedLabel[i] = getLabelMax(fkl);                                  //�����
		}
		
		double mif1 = Measures.microf1(this.uniqueLabels, test.y, predictedLabel);
		double maf1 = Measures.macrof1(this.uniqueLabels, test.y, predictedLabel);
		double hamm = Measures.averageSymLoss(test.y, predictedLabel);
		double zoloss = Measures.zeroOneLoss(test.y, predictedLabel);
		double[] perf = {mif1, maf1, hamm, zoloss};
		return perf;
	}
	
	/**
	 * @param testSample��������������k������
	 * @return Ԥ�����
	 */
	public double[] predictMax(Problem test, int k) {
		if (test == null) {
			return null;
		}
		
		double[][] pv = transTestSamples(test.x);
		int[][] predictedLabel = new int[test.l][];
		for (int i = 0; i < predictedLabel.length; i++) {
			double[] samplei = pv[i];
			double[] distance = new double[this.trainpv.length];
			for (int j = 0; j < this.trainpv.length; j++) {
				double[] sub = SparseVector.subVector(samplei, this.trainpv[j]);
				distance[j] = Math.sqrt(SparseVector.innerProduct(sub, sub));
			}
			int[] index = Sort.getIndexBeforeSort(distance);
			int[][] fkl = getFirstKlabel(this.extendy, index, (int)k, true);
			predictedLabel[i] = getLabelMax(fkl);
		}
		
		double mif1 = Measures.microf1(this.uniqueLabels, test.y, predictedLabel);
		double maf1 = Measures.macrof1(this.uniqueLabels, test.y, predictedLabel);
		double hamm = Measures.averageSymLoss(test.y, predictedLabel);
		double zoloss = Measures.zeroOneLoss(test.y, predictedLabel);
		double[] perf = {mif1, maf1, hamm, zoloss};
		return perf;
	}
	
	/**
	 * @param testSample��������������k������
	 * @return Ԥ�����
	 */
	public int[][] predictLabels(Problem test, int k) {
		if (test == null) {
			return null;
		}
		
		double[][] pv = transTestSamples(test.x);
		int[][] predictedLabel = new int[test.l][];
		for (int i = 0; i < predictedLabel.length; i++) {
			double[] samplei = pv[i];
			double[] distance = new double[this.trainpv.length];
			for (int j = 0; j < this.trainpv.length; j++) {
				double[] sub = SparseVector.subVector(samplei, this.trainpv[j]);
				distance[j] = Math.sqrt(SparseVector.innerProduct(sub, sub));
			}
			int[] index = Sort.getIndexBeforeSort(distance);
			int[][] fkl = getFirstKlabel(this.extendy, index, (int)k, true);
			predictedLabel[i] = getLabel(fkl);
		}
		return predictedLabel;
	}
	
	/**
	 * 
	 */
	private void getWeight() {
		if (this.prob == null) {
			return;
		}
		
		DataPoint[][] w = new DataPoint[this.uniqueLabels.length][];
		for (int i = 0; i < this.uniqueLabels.length; i++) {
			int label = this.uniqueLabels[i];
			int[] bl = getBinaryLabel(this.prob.y, label);
			double[] loss = new double[1];
			w[i] = Linear.train(this.prob, bl, this.param, null, loss, null, 0);
		}
		this.w = w;
	}
	
	/**
	 * 
	 */
	private double[][] transTestSamples(DataPoint[][] testSamples) {
		if (testSamples == null) {
			return null;
		}
		
		double[][] testPredictValues = new double[testSamples.length][this.uniqueLabels.length];
		for (int i = 0; i < testPredictValues.length; i++) {
			for (int j = 0; j < this.w.length; j++) {
				testPredictValues[i][j] = SparseVector.innerProduct(testSamples[i], this.w[j]);
			}
		}
		return testPredictValues;
	}
	
	/**
	 * �����������ÿһ��ģ����һ��Ϊ1
	 * @param mat��ά����
	 */
	private void scale(double[][] mat) {
		if (mat == null) {
			return;
		}
		
		for (int i = 0; i < mat.length; i++) {
			double inp = SparseVector.innerProduct(mat[i], mat[i]);
			double norm = Math.sqrt(inp);
			
			for (int j  = 0; j < mat[i].length; j++) {
				if (mat[i] != null && mat[i][j] != Double.NaN) {
					mat[i][j] = mat[i][j] / norm;
				} 
			}
		}
	}
}
