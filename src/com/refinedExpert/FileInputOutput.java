/**
 *读取训练样本相关操作 
 */
package com.refinedExpert;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class FileInputOutput {
	/**
	 * 适用于RCV1文件，读取文件ID，取得其对应类别。
	 * @param 
	 * @return 
	 * @throws IOException 
	 */
	public static Map<Integer, int[]>  getIDtoLabel(String qurels, String topics) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(qurels)));
		String line = null;
		Map<String, Integer> label2ID = getLabelID(topics);
		Map<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();
		while ((line = in.readLine()) != null) {
			String[] splits = line.split("\\s+");
			int key = Integer.parseInt(splits[1]);
			if (map.containsKey(key)) {
				List<Integer> value = map.get(key);
				int label = label2ID.get(splits[0]);
				value.add(label);
				map.put(key, value);
			} else {
				List<Integer> value = new ArrayList<Integer>();
				int label = label2ID.get(splits[0]);
				value.add(label);
				map.put(key, value);
			}
		}
		
		Map<Integer, int[]> rmap = new HashMap<Integer, int[]>();
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		while (it.hasNext()) {
			int key = it.next();
			List<Integer> value = map.get(key);
			int[] tvalue = new int[value.size()];
			for (int i = 0; i < tvalue.length; i++) {
				tvalue[i] = value.get(i);
			}
			rmap.put(key, tvalue);
		}
		return rmap;
	}
	
	/**
	 * 获得
	 */
	public static int[][] getLabel(int[][] y, Map<Integer, int[]> map) {
		if (y == null || map == null) {
			return null;
		}
		int[][] ry = new int[y.length][];
		for (int i = 0; i < ry.length; i++) {
			ry[i] = map.get(y[i][0]);
		}
		return ry;
	}
	
	/**
	 * 读取所有标签，并为每个标签指定数值标识。 
	 * @throws IOException 
	 */
	public static Map<String, Integer> getLabelID(String topics) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(topics)));
		List<String> topicList = new ArrayList<String>();
		String line = null;
		while ((line = in.readLine()) != null) {
			line = line.trim();
			topicList.add(line);
		}
		
		int id = 1;
		Map<String, Integer> labelIDmap = new HashMap<String, Integer>();
		for (int i = 0; i < topicList.size(); i++) {
			String key = topicList.get(i);
			int value = id;
			labelIDmap.put(key, value);
			id++;
		}
		return labelIDmap;
	}
}
