package com.refinedExpert;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Sort;

public class Tools {
	public static final int NONORMALIE = 0;
	public static final int HORIZONTAL = 1;
	public static final int VERTICAL = 2;
	/**
	 * @param y二维数组
	 * @return 返回数组中包含y中不重复的值，若y为null，返回null
	 */
	public static int[] getUniqueItem(int[][] y) {
		if (y == null) {
			return null;
		}
		Set<Integer> set = new HashSet<Integer>();
		for (int i = 0; i < y.length; i++) {
			for (int j = 0; j < y[i].length; j++) {
				set.add(y[i][j]);
			}
		}
		int[] uniqueItem = new int[set.size()];
		int index = 0;
		Iterator<Integer> it = set.iterator();
		while (it.hasNext()) {
			uniqueItem[index] = it.next();
			index++;
		}
		return uniqueItem;
	}
	
	/**
	 * @param train训练样本
	 * @param param这里是支持向量机参数
	 * @param uniqueLabels训练集包含的类标
	 */
	public static DataPoint[][] train(Problem train, Parameter param, int[] uniqueLabels) {
		if (train == null || param == null) {
			return null;
		}
		DataPoint[][] w = new DataPoint[uniqueLabels.length][];
		for (int i = 0; i < uniqueLabels.length; i++) {
			int label = uniqueLabels[i];
			int[] y = getBinaryLabels(train.y, label);
			double[] loss = new double[1];
			w[i] = Linear.train(train, y, param, null, loss, null, 0);
		}
		return w;
	}
	
	/**
	 * @param y输入数组
	 * @param num要查找的数
	 * @return 若y中包含num，返回true，否则返回false。
	 */
	public static boolean isContain(int[] y, int num) {
		if (y == null) {
			return false;
		}
		boolean contain = false;
		for (int i = 0; i < y.length; i++) {
			if (y[i] == num) {
				contain = true;
				break;
			} 
		}
		return contain;
	}
	
	/**
	 * @param labels样本对应类标，一行为一个样本对应的类标
	 * @param label指定类标
	 * @return 返回和labels行数相同的数组，内容为{1, -1}
	 */
	public static int[] getBinaryLabels(int[][] labels, int label) {
		if (labels == null) {
			return null;
		}
		int[] blabels = new int[labels.length];
		for (int i = 0; i < blabels.length; i++) {
			if (isContain(labels[i], label)) {
				blabels[i] = 1;
			} else {
				blabels[i] = -1;
			}
		}
		return blabels;
	}
	
	/**
	 * 
	 */
	public static double[][] transform(DataPoint[][] samples, DataPoint[][] weight, int normalizeType) {
		if (samples == null || weight == null) {
			return null;
		}
		int row = samples.length;
		int col = weight.length;
		double[][] tfMatrix = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				tfMatrix[i][j] = SparseVector.innerProduct(samples[i], weight[j]);
				if (tfMatrix[i][j] == Double.NEGATIVE_INFINITY 
						|| tfMatrix[i][j] == Double.POSITIVE_INFINITY
						|| tfMatrix[i][j] == Double.NaN) {
					System.out.println("Errors in Tools.transform, transformed matrix has infinity values.\n");
					return null;
				}
			}
		}
		
		if (normalizeType == Tools.HORIZONTAL) {
			horizontalNormalize(tfMatrix);
		}
		return tfMatrix;
	}
	
	/**
	 * @param mat输入二维矩阵，将矩阵每行模长归一化为1
	 */
	public static void horizontalNormalize(double[][] mat) {
		if (mat == null) {
			return;
		}
		
		for (int i = 0; i < mat.length; i++) {
			double sum = 0.0;
			for (int j = 0; j < mat[i].length; j++) {
				sum += mat[i][j] * mat[i][j];
			}
			sum = Math.sqrt(sum);
			for (int j = 0; j < mat[i].length; j++) {
				mat[i][j] = mat[i][j] / sum;
			}
		}
	}
	
	/**
	 * @param train, trainLabel, test, testLabel训练集和测试集样本及标签
	 * @param startFromZero近邻从0开始计算，multiLabel是否按多类标方式产生标签
	 * @return 返回测试性能，依次为microf1, macrof1, hamming loss, 0/1 loss
	 */
	public static double[] predict(double[][] train, int[][] trainLabel, double[][] test, int[][] testLabel, 
			int k, boolean startFromZero, boolean multiLabel) {
		if (train == null || trainLabel == null || test == null || testLabel == null) {
			return null;
		}
		
		int[][] predictedLabels = new int[test.length][];
		for (int i = 0; i < test.length; i++) {
			int[] index = getDistanceSort(train, test[i]);
			int[][] firstKlabel = getFirstKLabels(trainLabel, index, k, startFromZero);
			predictedLabels[i] = predict(firstKlabel, multiLabel);
		}
		
		int[] unique = getUniqueItem(trainLabel);
		double mif1 = Measures.microf1(unique, testLabel, predictedLabels);
		double maf1 = Measures.macrof1(unique, testLabel, predictedLabels);
		double hamloss = Measures.averageSymLoss(testLabel, predictedLabels);
		double zeroneloss = Measures.zeroOneLoss(testLabel, predictedLabels);
		double[] performance = {mif1, maf1, hamloss, zeroneloss};
		return performance;
	}
	
	/**
	 * @param train, trainLabel, test, testLabel训练集和测试集样本及标签
	 * @param startFromZero近邻从0开始计算，multiLabel是否按多类标方式产生标签
	 * @return 返回测试性能，依次为microf1, macrof1, hamming loss, 0/1 loss
	 */
	public static double[][] predict(double[][] train, int[][] trainLabel, double[][] test, int[][] testLabel, 
			int[] k, boolean startFromZero, boolean multiLabel) {
		double[][] perfs = new double[k.length][];
		for (int i = 0; i < k.length; i++) {
			int tk = k[i];
			perfs[i] = predict(train, trainLabel, test, testLabel, tk, startFromZero, multiLabel);
		}
		return perfs;
	}
	
	/**
	 * 
	 */
	private static int[] predict(int[][] y, boolean multiLabel) {
		if (y == null) {
			return null;
		}
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < y.length; i++) {
			for (int j = 0; j < y[i].length; j++) {
				int key = y[i][j];
				if (map.containsKey(key)) {
					int value = map.get(key);
					value = value + 1;
					map.put(key, value);
				} else {
					map.put(key, 1);
				}
			}
		}
		
		int[] pl = null;
		if (multiLabel) {
			double threshold = ((double)y.length) / 2;
			List<Integer> list = new ArrayList<Integer>();
			Set<Integer> keySet = map.keySet();
			Iterator<Integer> it = keySet.iterator();
			while (it.hasNext()) {
				int key = it.next();
				int value = map.get(key);
				if (value > threshold) {
					list.add(key);
				}
			}
			int[] mpl = new int[list.size()];
			for (int i = 0; i < mpl.length; i++) {
				mpl[i] = list.get(i);
			}
			pl = mpl;
		} else {
			Set<Integer> keySet = map.keySet();
			Iterator<Integer> it = keySet.iterator();
			int max = Integer.MIN_VALUE;
			int label = -1;
			while (it.hasNext()) {
				int key = it.next();
				int value = map.get(key);
				if (value > max) {
					label = key;
				}
			}
			int[] spl = {label};
			pl = spl;
		}
		return pl;
	}
	
	/**
	 * 
	 */
	private static int[][] getFirstKLabels(int[][] y, int[] index, int k, boolean startFromZero) {
		if (y == null || index == null) {
			return null;
		}
		int[][] fkl = new int[k][];
		int base;
		if (startFromZero) {
			base = 0;
		} else {
			base = 1;
		}
		for (int i = 0;  i < k; i++) {
			fkl[i] = y[index[base+i]];
		}
		return fkl;
	}
	
	/**
	 * 
	 */
	public static int[] getDistanceSort(double[][] train, double[] test) {
		if (train == null || test == null) {
			return null;
		}
		double[] distance = new double[train.length];
		for (int i = 0; i < distance.length; i++) {
			double[] sub = SparseVector.subVector(train[i], test);
			double dis2 = SparseVector.innerProduct(sub, sub);
			distance[i] = dis2;
		}
		int[] index = Sort.getIndexBeforeSort(distance);
		return index;
	}
}
