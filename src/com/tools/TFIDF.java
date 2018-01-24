package com.tools;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

public class TFIDF {
	/**
	 * 
	 */
	public static Map<Integer, Double> idf(Problem prob) {
		Map<Integer, Double> map = new HashMap<Integer, Double>();
		DataPoint dp = null;
		int key;
		double value;
		for(int i = 0; i < prob.l; i++) {
			for(int j = 0; j < prob.x[i].length; j++) {
				dp = prob.x[i][j];
				key = dp.index;
				if(map.containsKey(key)) {
					value = map.get(key);
					value = value + 1;
					map.put(key, value);
				} else {
					map.put(key, 1.0);
				}
			}
		}
		
		double n = prob.l;
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			value = Math.log(n / value);
			map.put(key, value);
		}
		return map;
	}
	
	/**
	 * 
	 */
	public static void tfidf(Problem prob) {
		Map<Integer, Double> map = idf(prob);
		int key;
		double value;
		for(int i = 0; i < prob.l; i++) {
			for(int j = 0; j < prob.x[i].length; j++) {
				key = prob.x[i][j].index;
				value = prob.x[i][j].value;
				value = value * map.get(key);
				prob.x[i][j].value = value;
			}
		}
	}
	
	public static void tfidf(Problem prob, Map<Integer, Double> map) {
		int key;
		double value;
		for(int i = 0; i < prob.l; i++) {
			for(int j = 0; j < prob.x[i].length; j++) {
				key = prob.x[i][j].index;
				value = prob.x[i][j].value;
				if(map.get(key) == null) {
					value = 0;
				} else {
					value = value * map.get(key);
				}
				prob.x[i][j].value = value;
			}
		}
	}
	
	/**
	 * ltc 
	 */
	public static void ltc(Problem prob, Map<Integer, Double> map) {
		int key;
		double value;
		double sum = 0;
		for(int i = 0; i < prob.l; i++) {
			sum = 0;
			for(int j = 0; j < prob.x[i].length; j++) {
				key = prob.x[i][j].index;
				value = prob.x[i][j].value;
				if(map.get(key) == null) {
					value = 0;
				} else {
					value = (1 + Math.log(value)) * map.get(key);
				}
				prob.x[i][j].value = value;
				sum += value * value;
			}
			
			sum = Math.pow(sum, 0.5);
			for(int j = 0; j < prob.x[i].length; j++) {
				prob.x[i][j].value /= sum;
			}
		}
		
	}
	
	/**
	 * 
	 */
	public static void scale(Problem prob) {
		double sum = 0;
		for(int i = 0; i < prob.l; i++) {
			sum = 0;
			for(int j = 0; j < prob.x[i].length; j++) {
				sum += prob.x[i][j].value * prob.x[i][j].value;
			}
			for(int j = 0; j < prob.x[i].length; j++) {
				prob.x[i][j].value = prob.x[i][j].value / Math.pow(sum, 0.5);
			}
		}
	}
	
	/**
	 * 添加一维特征向量 
	 */
	public static void extendProblem(Problem prob) {
		int index = prob.n + 1;
		DataPoint[] dp = null;
		for(int i = 0; i < prob.l; i++) {
			dp = new DataPoint[prob.x[i].length + 1];
			int j = 0;
			for(j = 0; j < prob.x[i].length; j++) {
				dp[j] = new DataPoint(prob.x[i][j].index, prob.x[i][j].value);
			}
			dp[j] = new DataPoint(index, 1);
			prob.x[i] = dp;
		}
	}
	
	/**
	 * 
	 */
	public static void norm(Problem prob) {
		for(int i = 0; i < prob.l; i++) {
			double sum = 0;
			for(int j = 0; j < prob.x[i].length; j++) {
				sum += prob.x[i][j].value * prob.x[i][j].value;
			}
			System.out.println(sum);
		}
	}
	
	/**
	 * document frequency
	 */
	public static Map<Integer, Integer> df(Problem prob) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int key;
		int value;
		for(int i = 0; i < prob.l; i++) {
			for(int j = 0; j < prob.x[i].length; j++) {
				key = prob.x[i][j].index;
				if(map.containsKey(key)) {
					value = map.get(key);
					value = value + 1;
					map.put(key, value);
				} else {
					map.put(key, 1);
				}
			}
		}
		System.out.println("Different items " + map.size());
		return map;
	}
	
	/**
	 * 返回n * 2矩阵，第一列为item， 第二列为文档频率
	 */
	public static int[][] itemDf(Map<Integer, Integer> map) {
		int[][] df = new int[map.size()][];
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		
		int key;
		int value;
		
		int index = 0;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			
			df[index][0] = key;
			df[index][1] = value;
			
			index = index + 1;
		}
		return df;
	}
}
