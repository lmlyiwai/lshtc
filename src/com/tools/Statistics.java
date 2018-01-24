package com.tools;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class Statistics {
	/**
	 * 统计id在y中出现多少次
	 * */
	public static int getTotleNums(int[][] y, int id) {
		int result = 0;
		if(y == null) {
			return result;
		}
		for(int i = 0; i < y.length; i++) {
			if(Contain.contain(y[i], id)) {
				result++;
			}
		}
		
		return result;
	}

	/**
	 *	统计样本总类别数
	 */
	public static int[] getUniqueLabels(int[][] y) {
		if(y == null) {
			return null;
		}
		
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
	 * 将出现的类标组合作为新类标，并统计每类出现次数 
	 */
	public static Map<int[], Integer> multiLabelStatistics(int[][] y) {
		Map<String, Integer> map = new HashMap<String, Integer>();
		
		int[] temp;
		String str;
		for(int i = 0; i < y.length; i++) {
			temp = Sort.sort(y[i]);
			str = new String();
			for(int j = 0; j < temp.length; j++) {
				str += temp[j] + " ";
			}
			
			if(map.containsKey(str)) {
				int value = map.get(str);
				value = value + 1;
				map.put(str, value);
			} else {
				map.put(str, 1);
			}
		}
		
		Map<int[], Integer> result = new HashMap<int[], Integer>();
		Set<String> set = map.keySet();
		Iterator<String> it = set.iterator();
		
		String[] splits = null;
		while(it.hasNext()) {
			str = it.next();
			int value = map.get(str);
			
			splits = str.split("\\s+");
			
			temp = new int[splits.length];
			for(int i = 0; i < temp.length; i++) {
				temp[i] = Integer.parseInt(splits[i]);
			}
			
			result.put(temp, value);
		}
		
		return result;
	}
	
	/**
	 * 
	 */
	public static int[][] getLabelsStatistic(int[][] y) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int key;
		int value;
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				key = y[i][j];
				if(map.containsKey(key)) {
					value = map.get(key);
					value = value + 1;
					map.put(key, value);
				} else {
					map.put(key, 1);
				}
			}
		}
		
		int[][] stat = new int[2][map.size()];
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		
		int counter = 0;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			stat[0][counter] = key;
			stat[1][counter] = value;
			counter = counter + 1;
		}
		return stat;
	}
}
