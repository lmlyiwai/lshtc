package com.IMCLEF;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.tools.Contain;

public class ProcessIMCLEF {
	/**
	 * 读取目录结构 
	 * @throws IOException 
	 */
	public static Map<String, Integer> readStructure(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		Set<String> set = new HashSet<String>();
		
		String line = null;
		String[] splits = null;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\r|\n|\t");
			if(splits != null) {
				for(int i = 0; i < splits.length; i++) {
					String str = splits[i].trim();
					set.add(str);
				}
			}
		}
		
		Iterator<String> it = set.iterator();
		int id = 0;
		Map<String, Integer> map = new HashMap<String, Integer>();
		while(it.hasNext()) {
			String key = it.next();
			map.put(key, id++);
		}
		
		in.close();
		return map;
	}
	
	/**
	 * @throws IOException 
	 *	 
	 */
	public static int[][] edge(String filename, Map<String, Integer> map) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		
		List<Integer> parent = new ArrayList<Integer>();
		List<Integer> child = new ArrayList<Integer>();
		
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\t|\n|\r");
			if(splits != null) {
				String strp = splits[0].trim();
				String strc = splits[1].trim();
				
				int pid = map.get(strp);
				int cid = map.get(strc);
				
				parent.add(pid);
				child.add(cid);
			}
		}
		
		int[][] result = new int[parent.size()][2];
		for(int i = 0; i < result.length; i++) {
			result[i][0] = parent.get(i);
			result[i][1] = child.get(i);
		}
		
		in.close();
		return result;
	}
	
	/**
	 *	key value 调换 
	 */
	public static Map<Integer, String> reverseMap(Map<String, Integer> map) {
		Map<Integer, String> result = new HashMap<Integer, String>();
		
		Set<String> set = map.keySet();
		Iterator<String> it = set.iterator();
		
		String key;
		int value;
		
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			
			result.put(value, key);
		}
		
		return result;
	}
	
	/**
	 * 类标转化为只与叶节点关联
	 */
	public static int[][] pathToLeaf(int[][] y, int[] leaves) {
		int[][] result = new int[y.length][];
		
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < y.length; i++) {
			list.clear();
			for(int j = 0; j < y[i].length; j++) {
				if(Contain.contain(leaves, y[i][j])) {
					list.add(y[i][j]);
				}
			}
			
			result[i] = new int[list.size()];
			for(int j = 0; j < result[i].length; j++) {
				result[i][j] = list.get(j);
			}
		}
		return result;
	}
}
