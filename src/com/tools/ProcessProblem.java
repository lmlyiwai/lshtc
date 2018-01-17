package com.tools;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class ProcessProblem {
	/**
	 * 获得样本集类标 
	 */
	public static int[] getUniqueLabels(int[][] y) {
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				set.add(y[i][j]);
			}
		}
		
		int[] result = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		int counter = 0;
		while(it.hasNext()) {
			result[counter++] = it.next();
		}
		return result;
	}
	
	/**
	 * 类标转化为矩阵形式 
	 */
	public static double[][] labelToMatrix(int[][] y, int[] ulables) {
		double[][] result = new double[y.length][ulables.length];
		for(int i = 0; i < y.length; i++) {
			int[] ty = y[i];
			for(int j = 0; j < ty.length; j++) {
				for(int k = 0; k < ulables.length; k++) {
					if(ty[j] == ulables[k]) {
						result[i][k] = 1;
					}
				}
			}
		}
		return result;
	}
}
