package com.simulate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class Tools {
	/**
	 * 
	 */
	public static int[][] getClassIndex(int[][] y) {
		String[] labels = getLabelSet(y);
		int[] index= new int[labels.length];         //
		
		int[] labelsSum = new int[labels.length];    //每一类的样本数目
		
		String line = null;
		for(int i = 0; i < y.length; i++) {
			Arrays.sort(y[i]);
			line = new String();
			for(int j = 0; j < y[i].length; j++) {
				line = line + y[i][j] + " ";
			}
			
			for(int j = 0; j < labels.length; j++) {
				if(line.equals(labels[j])) {
					labelsSum[j]++;
				}
			}
		}
		
		int[][] result = new int[labels.length][];
		for(int i = 0; i < labels.length; i++) {
			result[i] = new int[labelsSum[i]];
		}
		
		for(int i = 0; i < y.length; i++) {
			line = new String();
			for(int j = 0; j < y[i].length; j++) {
				line = line + y[i][j] + " ";
			}
			
			for(int j = 0; j < labels.length; j++) {
				if(line.equals(labels[j])) {
					result[j][index[j]] = i;
					index[j] = index[j] + 1;
				}
			}
		}
		
		return result;
	}
	
	/**
	 * 获得所有类标，类标子集作为一个类
	 */
	public static String[] getLabelSet(int[][] y) {
		Set<String> set = new HashSet<String>();
		String line = null;
		for(int i = 0; i < y.length; i++) {
			Arrays.sort(y[i]);
			line = new String();
			for(int j = 0; j < y[i].length; j++) {
				line = line + y[i][j] + " ";
			}
			set.add(line);
		}
		
		String[] labels = new String[set.size()];
		Iterator<String> it = set.iterator();
		int counter = 0;
		while(it.hasNext()) {
			labels[counter++] = it.next();
		}
		return labels;
	}
	
	/**
	 * 取2/3训练集，1/3确认集 
	 */
	public static int[][] splits(int[][] index) {
		List<Integer> trainList = new ArrayList<Integer>();
		List<Integer> validList = new ArrayList<Integer>();
		for(int i = 0; i < index.length; i++) {
			int mid = (int)Math.round(index[i].length * 0.66);
			for(int j = 0; j < mid; j++) {
				trainList.add(index[i][j]);
			}
			for(int j = mid; j < index[i].length; j++) {
				validList.add(index[i][j]);
			}
		}
		
		int[][] result = new int[2][];
		result[0] = new int[trainList.size()];
		result[1] = new int[validList.size()];
		for(int i = 0; i < result[0].length; i++) {
			result[0][i] = trainList.get(i);
		}
		for(int i = 0; i < result[1].length; i++) {
			result[1][i] = validList.get(i);
		}
		return result;
	}
}
