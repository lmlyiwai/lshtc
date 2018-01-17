package com.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class FileInputOutput {
	
	//读取叶节点类标
	public static String[] readLabels(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		List<String> labels = new ArrayList<String>();
		String line = null;
		while((line = in.readLine()) != null) {
			line = line.toUpperCase().trim();
			labels.add(line);
		}
		in.close();
		
		String[] result = new String[labels.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = labels.get(i);
		}
		return result;
		
	}

	//读取节点
	public static String[][] readLabelPairs(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		List<String> parentList = new ArrayList<String>();
		List<String> childList = new ArrayList<String>();
		
		String[] splits = null;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s|:\\s|\t|\t|\n");
			if(!splits[1].equals("None")) {
				parentList.add(splits[1].toUpperCase());
				childList.add(splits[3].toUpperCase());
			}
		}
		
		String[][] result = new String[parentList.size()][2];
		for(int i = 0; i < parentList.size(); i++) {
			result[i][0] = parentList.get(i);
			result[i][1] = childList.get(i);
		}
		return result;
	} 

	//给每个类别指定一个数字ID
	public static Map<String, Integer> getIDLabelPair(String[][] pairs) {
		Set<String> set = new HashSet<String>();
		for(int i = 0; i < pairs.length; i++) {
			set.add(pairs[i][0]);
			set.add(pairs[i][1]);
		}
		
		Map<String, Integer> map = new HashMap<String, Integer>();
		
		Iterator<String> it = set.iterator();
		String str;
		int counter = 0;
		while(it.hasNext()) {
			str = it.next();
			map.put(str, counter++);
		}
		return map;
	}

	//
	public static void showMap(Map<String, Integer> map) {
		Set<String> set = map.keySet();
		Iterator<String> it = set.iterator();
		
		String str;
		int id;
		while(it.hasNext()) {
			str = it.next();
			id = map.get(str);
			System.out.println(str + " --> " + id);
		}
	}

	//父子关系，用数值表示
	public static int[][] labelPairs(String[][] result, Map<String, Integer> map) {
		int rows = result.length;
		int cols = result[0].length;
		int[][] pc = new int[rows][cols];
		
		for(int i = 0; i < rows; i++) {
			pc[i][0] = map.get(result[i][0]);
			pc[i][1] = map.get(result[i][1]);
		}
		return pc;
	}

	//获得给定文本的类别
	public static Map<Integer, int[]> getDocLabel(String filename, Map<String, Integer> map) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		
		Map<Integer, int[]> m = new HashMap<Integer, int[]>();
		
		String fileID = null;
		String lastFileID = null;
		int[] labels = null;
		List<Integer> list = new ArrayList<Integer>();
		while((line = in.readLine()) != null) {
			splits = line.split("\\s|\r|\t|\n");
			fileID = splits[1];
			if(lastFileID == null || fileID.equals(lastFileID)) {
				Integer tid = map.get(splits[0]);
				if(tid != null) {
					list.add(tid);	
				}
			} else {
//System.out.println("file id = " + fileID + ", list = " + list + ", list size = " + list.size());
				labels = new int[list.size()];
				for(int i = 0; i < labels.length; i++) {
					if(list.get(i) != null) {
						labels[i] = list.get(i);
					}
				}
				m.put(Integer.parseInt(lastFileID), labels);
				list.clear();
				if(map.get(splits[0]) != null) {
					list.add(map.get(splits[0]));
				}
			}
			lastFileID = fileID;
		}
		
		labels = new int[list.size()];
		for(int i = 0; i < labels.length; i++) {
			labels[i] = list.get(i);
		}
		m.put(Integer.parseInt(lastFileID), labels);
		list.clear();
		return m;
	}
	
	//
	public static Map<Integer, String> reverse(Map<String, Integer> map) {
		Map<Integer, String> rev = new HashMap<Integer, String>();
		
		Set<String> key = map.keySet();
		Iterator<String> it = key.iterator();
		
		String str;
		int value;
		while(it.hasNext()) {
			str = it.next();
			value = map.get(str);
			rev.put(value, str);
		}
		
		return rev;
	}
	
	/**
	 * @throws IOException 
	 * 
	 * */
	public static void writeVectorToFile(String filename, DataPoint[][] w) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		
		DataPoint[] tw = null;
		DataPoint dp = null;
		String line;
		for(int i = 0; i < w.length; i++) {
			tw = w[i];
			line = new String();
			if(tw == null) {
				continue;
			}
			for(int j = 0; j < tw.length; j++) {
				dp = tw[j];
				line = line + dp.index + ":" + dp.value + " ";
			}
			line = line + "\n";
			out.write(line);
		}
		out.flush();
		out.close();
	}

	/**
	 * @throws IOException 
	 * 
	 * */
	public static DataPoint[][] readVertorFromFile(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		List<DataPoint[]> list = new ArrayList<DataPoint[]>();
		String line = null;
		int m;
		DataPoint[] temp = null;
		String[] splits = null;
		while((line = in.readLine()) != null) {
			line = line.trim();
			splits = line.split("\\s+|:");
			m = splits.length / 2;
			temp = new DataPoint[m];
			for(int j = 0; j < m; j++) {
				temp[j] = new DataPoint(Integer.parseInt(splits[2 * j]), Double.parseDouble(splits[2 * j + 1]));
			}
			list.add(temp);
		}
		in.close();
		
		DataPoint[][] result = new DataPoint[list.size()][];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
		
	}
	
	public static void writeArrayToFile(String filename, int[][] y) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		int[] temp;
		String line = null;
		for(int i = 0; i < y.length; i++) {
			temp = y[i];
			line = new String();
			for(int j = 0; j < temp.length; j++) {
				line += temp[j] + " ";
			}
			line += "\n";
			out.write(line);
		}
		out.close();
	}
	
	/**
	 *将原来可能包含中间节点的类标转变成是包含叶节点的类标 	
	 *map中是为中间节点添加的父子节点
	 * */
	public static int[][] oldTransLabels(int[][] y, Structure tree, Map<Integer, Integer> map) {
		int[][] result = new int[y.length][];
		Set<Integer> innerSet = map.keySet();
		Set<Integer> leaveSet = new HashSet<Integer>();
		Iterator<Integer> it = innerSet.iterator();
		
		int pid;
		while(it.hasNext()) {
			pid = it.next();
			leaveSet.add(map.get(pid));
		}
		
		List<Integer> list = new ArrayList<Integer>();
		int[] ty = null;
		int[] children = null;
		for(int i = 0; i < y.length; i++) {
			ty = y[i];
			list.clear();
			for(int j = 0; j < ty.length; j++) {
				if(tree.isLeaf(ty[j])) {
					list.add(ty[j]);
				} else{
					children = tree.getChildren(ty[j]);
					boolean add = true;
					for(int k = 0; k < children.length; k++) {
						if(Contain.contain(ty, children[k])) {
							add = false;
							break;
						}
					}
					if(add) {
						if(map.get(ty[j]) != null) {
							list.add(map.get(ty[j]));
						}
					}
				}
			}
			
			int[] tl = new int[list.size()];
			for(int m = 0; m < tl.length; m++) {
				tl[m] = list.get(m);
			}
			result[i] = tl;
		}
		return result;
	}
	
	public static int[][] transLabels(int[][] y, Structure tree, Map<Integer, Integer> map) {
		int[][] result = new int[y.length][];
		int[] tempLabel = null;
		int i;
		int j;
		int id;
		for(i = 0; i < y.length; i++) {
			tempLabel = new int[y[i].length];
			for(j = 0; j < y[i].length; j++) {
				id = y[i][j];
				if(tree.isLeaf(id)) {
					tempLabel[j] = id;
				} else {
					tempLabel[j] = map.get(id);
				}
			}
			result[i] = tempLabel;
		}
		return result;
	}

	public static int[][] extendLabelToOld(Structure struc, int[][] y) {
		Map<Integer, Integer> innerToleaf = struc.getInnerToAdd();
		Map<Integer, Integer> leafToInnter = newReverse(innerToleaf);
		
		int[][] result = new int[y.length][];
		for(int i = 0; i < y.length; i++) {
			
			result[i] = new int[y[i].length];
			
			for(int j = 0; j < y[i].length; j++) {
				if(leafToInnter.get(y[i][j]) == null) {
					result[i][j] = y[i][j];
				} else {
					result[i][j] = leafToInnter.get(y[i][j]);
				}
			}
		}
		
		return result;
	}
	
	public static Map<Integer, Integer> newReverse(Map<Integer, Integer> map) {
		Map<Integer, Integer> rev = new HashMap<Integer, Integer>();
		
		Set<Integer> key = map.keySet();
		Iterator<Integer> it = key.iterator();
		
		int k;
		int value;
		while(it.hasNext()) {
			k = it.next();
			value = map.get(k);
			rev.put(value, k);
		}
		
		return rev;
	}
	
	/**
	 * 将ID对应的类别写入文件
	 * @throws IOException 
	 * */
	public static void writeIDLabelToFile(String filename, Map<String, Integer> map) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		
		Set<String> set = map.keySet();
		Iterator<String> it = set.iterator();
		
		String 	key;
		int 	value;
		String 	line;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			line = key + " " + value + "\n";
			out.write(line);
		}
		out.close();
	}
	
	public static void writeExtendIDLabelToFile(String filename, Map<String, Integer> map, Map<Integer, Integer> oldTonew) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		Set<String> set = map.keySet();
		Iterator<String> it = set.iterator();
		
		String 	key;
		int 	value;
		String 	line;
		
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			if(oldTonew.containsKey(value)) {
				value = oldTonew.get(value);
			}
			
			line = key + " " + value + "\n";
			out.write(line);
		}
		out.close();
	}
	
	public static void writeMapToFile(String filename, Map<Integer, Integer> map) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		
		String line = null;
		int label;
		int totle;
		while(it.hasNext()) {
			line = new String();
			label = it.next();
			totle = map.get(label);
			line = label + " - " + totle + "\n";
			out.write(line);
		}
		out.flush();
		out.close();
	}

	public static void writeMapToFile(Map<Integer, Integer> map, String filename) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(filename)));
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		int key;
		int value;
		String line = null;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			line = new String();
			line = key + " -- " + value + "\n";
			out.write(line);
		}
		out.close();
	}
	
	public static void writeProblemToFile(String filename, Problem prob) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(filename)));
		int i, j;
		String line = null;
		for(i = 0; i < prob.l; i++) {
			line = new String();
			for(j = 0; j < prob.y[i].length; j++) {
				if(j < prob.y[i].length - 1) {
					line += prob.y[i][j] + ",";
				} else {
					line += prob.y[i][j] + " ";
				}
			}
			
			for(j = 0; j < prob.x[i].length; j++) {
				line += prob.x[i][j].index + ":" + prob.x[i][j].value + " ";
			}
			line += "\n";
			out.write(line);
		}
		out.close();
	}
}
