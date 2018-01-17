package com.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class CrossValidation {
	/**
	 * 同一类样本下标放一起
	 * */
	public static int[][] getEachLabel(int[][] y) {
		int[] temp;
		Map<String, Integer> map = new HashMap<String, Integer>();
		int i, j;
		String labels = null; 
		String key;
		int value;
		String[] strlabels = new String[y.length];
		
		for(i = 0; i < y.length; i++) {
			temp = new int[y[i].length];
			for(j = 0; j < y[i].length; j++) {
				temp[j] = y[i][j];
			}
			
			Arrays.sort(temp);
			labels = new String();
			for(j = 0; j < temp.length; j++) {
				labels += temp[j];
			}
			
			strlabels[i] = labels;
			
			if(map.containsKey(labels)) {
				value = map.get(labels);
				map.put(labels, value+1);
			} else {
				map.put(labels, 1);
			}
		}
		
		
		int[][] result = new int[map.size()][];
		
		Set<String> set = map.keySet();
		Iterator<String> it = set.iterator();
		i = 0;
		int index;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			result[i] = new int[value];
			index = 0;
			for(j = 0; j < strlabels.length; j++) {
				if(strlabels[j].equals(key)) {
					result[i][index++] = j;
				}
			}
			i++;
		}
		return result;
	}

	/**
	 * 获得训练样本，确认样本
	 * */
	public static int[][] getTrainValidIndex(int[][] indexs, int n_fold, int it) {
		int[][] result = new int[2][];
		
		List<Integer> trainIndex = new ArrayList<Integer>();
		List<Integer> validIndex = new ArrayList<Integer>();
		int i;
		int[] temp;
		for(i = 0; i < indexs.length; i++) {
			temp = indexs[i];
			if(temp.length == 1) {
				trainIndex.add(temp[0]);
//				validIndex.add(temp[0]);
			} else if(temp.length < 10) {
				double n = Math.ceil(temp.length * 0.3);
				boolean[] flag = new boolean[temp.length];
				int k;
				for(k = 0; k < flag.length; k++) {
					flag[k] = false;
				}
				for(k = 0; k < n; k++) {
					int t = (int)(Math.random() * temp.length);
					flag[t] = true;
				}
				for(k = 0; k < temp.length; k++) {
					if(flag[k]) {
						validIndex.add(temp[k]);
					} else {
						trainIndex.add(temp[k]);
					}
				}
			} else {
				int length = temp.length / n_fold;
				int start = it * length;
				int end = start + length;
				int k;
				for(k = start; k < end; k++) {
					validIndex.add(temp[k]);
				}
				for(k = 0; k < start; k++) {
					trainIndex.add(temp[k]);
				}
				for(k = end; k < temp.length; k++) {
					trainIndex.add(temp[k]);
				}
			}
		}
		
		result[0] = new int[trainIndex.size()];
		result[1] = new int[validIndex.size()];
		
		for(i = 0; i < result[0].length; i++) {
			result[0][i] = trainIndex.get(i);
		}
		for(i = 0; i < result[1].length; i++) {
			result[1][i] = validIndex.get(i);
		}
		return result;
	}
}
