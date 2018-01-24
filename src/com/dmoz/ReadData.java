package com.dmoz;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class ReadData {
	public static int[][] readLabelPair(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		
		List<String> parent = new ArrayList<String>();
		List<String> child = new ArrayList<String>();
		
		String line;
		String[] splits;
		
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\r|\n");
			parent.add(splits[0].trim());
			child.add(splits[1].trim());
		}
		
		int[][] result = new int[parent.size()][2];
		
		for(int i = 0; i < result.length; i++) {
			result[i][0] = Integer.parseInt(parent.get(i));
			result[i][1] = Integer.parseInt(child.get(i));
		}
		
		in.close();
		
		return result;
	}
	
	/**
	 * 	LSHTC-small,读类别标签
	 * @throws IOException 
	 * */
	public static String[] readPathToLeaf(String filename) throws IOException {
		BufferedReader in = null;
		
		List<String> list = new ArrayList<String>();
		String line = null;
		
		try {
			in = new BufferedReader(new InputStreamReader(
					new FileInputStream(new File(filename))));
			
			while((line = in.readLine()) != null) {
				line.trim();
				list.add(line);
			}
		} finally {
			in.close();
		}
		
		String[] result = new String[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}

	/**
	 * 	获得所有类别编号
	 * */
	public static int[] getAllLabels(String[] paths) {
		Set<Integer> set = new HashSet<Integer>();
		String line = null;
		String[] splits = null;
		int j;
		for(int i = 0; i < paths.length; i++) {
			line = paths[i];
			splits = line.split("\\s+|\r|\n");
			for(j = 0; j < splits.length; j++) {
				set.add(Integer.parseInt(splits[j]));
			}
		}
		
		int[] result = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		
		j = 0;
		while(it.hasNext()) {
			result[j++] = it.next();
		}
		
		return result;
	}
	
	/**
	 * 给每个类别指定一个编号
	 * */
	public static Map<Integer, Integer> getLabelID(int[] labels) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int counter = 1;
		for(int i = 0; i < labels.length; i++) {
			map.put(labels[i], counter++);
		}
		return map;
	}
	
	/**
	 * 获得结构中子树的根
	 * */
	public static int[] getForestRoot(String[] paths) {
		Set<Integer> set = new HashSet<Integer>();
		String[] splits = null;
		for(int i = 0; i < paths.length; i++) {
			splits = paths[i].split("\\s+");
			set.add(Integer.parseInt(splits[0]));
		}
		
		int[] result = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		int i = 0;
		while(it.hasNext()) {
			result[i++] = it.next();
		}
		
		return result;
	}
	
	/**
	 * 构建树
	 * */
	public static void setupStructure(Structure stru, int[] roots, String[] paths, Map<Integer, Integer> map) {
		int root = 0;
		int i,j;
		String line = null;
		String[] splits = null;
		
		for(i = 0; i < roots.length; i++) {
			stru.addChild(root, map.get(roots[i]));
		}
		
		int parent, child;
		for(i = 0; i < paths.length; i++) {
			line = paths[i];
			splits = line.split("\\s+|\r|\n");
			for(j = 0; j < splits.length - 1; j++) {
				parent = Integer.parseInt(splits[j]);
				child = Integer.parseInt(splits[j + 1]);
				
				parent = map.get(parent);
				child = map.get(child);
				
				stru.addChild(parent,  child);
			}
		}
	}
	
	public static Map<Integer, Integer> reverseMap(Map<Integer, Integer> map) {
		Map<Integer, Integer> result = new HashMap<Integer, Integer>();
		int key, value;
		
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			
			result.put(value, key);
		}
		
		return result;
	}
	
	/**
	 * 读入训练样本，除去index 0
	 * */
	public static Problem readProblem(File file, double bias) throws IOException, InvalidInputDataException {
		BufferedReader fp = new BufferedReader(new FileReader(file));
		List<int[]> vy = new ArrayList<int[]>();
		List<DataPoint[]> vx = new ArrayList<DataPoint[]>();
		
		int max_index = 0;
		
		int lineNr = 0;
		
		try {
			while(true) {
				String line = fp.readLine();
				if(line == null) break;
				lineNr++;
				
				String[] st = line.split("\\s+|\t|\n|\r|\f|:");
				if(st.length <= 1) {
					System.out.println("检查输入样本格式");
					return null;
				}
				
				String label = st[0];					
				String[] labels = label.split(",");
				int[] labs = new int[labels.length];		//样本对应标签
				for(int i = 0; i < labs.length; i++) {
					labs[i] = Integer.parseInt(labels[i]);
				}
				vy.add(labs);
				
				int m = st.length / 2 - 1;
				DataPoint[] x;
				if(bias >= 0) {
					x = new DataPoint[m+1];
				} else {
					x = new DataPoint[m];
				}
				
				String token;
				for(int j = 1; j < m+1; j++) {
					token = st[2 * j + 1];
					token = token.trim();
					int index;
					try {
						index = Integer.parseInt(token);
					} catch (NumberFormatException e) {
						throw new InvalidInputDataException("无效的index:" + token, file, lineNr, e);
					}
					
					if(index < 0) throw new InvalidInputDataException("无效的index:"+index, file, lineNr);
					
				
					token  = st[2 * j + 2];
					try {
						double value = Double.parseDouble(token);
						x[j - 1] = new DataPoint(index, value);
					} catch (NumberFormatException e) {
						throw new InvalidInputDataException("无效的value:" + token, file, lineNr);
					}
					
					max_index = Math.max(max_index, x[j-1].index);
				}
	
				vx.add(x);
			}
			
			return constructProblem(vy, vx, max_index, bias);
		} finally {
			fp.close();
		}
	}

	/**
	 * 读入样本，样本中不包含index 0
	 * */
	public static Problem newReadProblem(File file, double bias) throws IOException, InvalidInputDataException {
		BufferedReader fp = new BufferedReader(new FileReader(file));
		List<int[]> vy = new ArrayList<int[]>();
		List<DataPoint[]> vx = new ArrayList<DataPoint[]>();
		
		int max_index = 0;
		
		int lineNr = 0;
		
		try {
			while(true) {
				String line = fp.readLine();
				if(line == null) break;
				lineNr++;
				
				String[] st = line.split("\\s+|\t|\n|\r|\f|:");
				if(st.length <= 1) {
					System.out.println("检查输入样本格式");
					return null;
				}
				
				String label = st[0];					
				String[] labels = label.split(",");
				int[] labs = new int[labels.length];		//样本对应标签
				for(int i = 0; i < labs.length; i++) {
					labs[i] = Integer.parseInt(labels[i].trim());
				}
				vy.add(labs);
				
				int m = st.length / 2;
				DataPoint[] x;
				if(bias >= 0) {
					x = new DataPoint[m+1];
				} else {
					x = new DataPoint[m];
				}
				
				String token;
				for(int j = 0; j < m; j++) {
					token = st[2 * j + 1];
					token = token.trim();
					int index;
					try {
						index = Integer.parseInt(token);
					} catch (NumberFormatException e) {
						throw new InvalidInputDataException("无效的index:" + token, file, lineNr, e);
					}
					
					if(index < 0) throw new InvalidInputDataException("无效的index:"+index, file, lineNr);
					
				
					token  = st[2 * j + 2];
					try {
						double value = Double.parseDouble(token);
						x[j] = new DataPoint(index, value);
					} catch (NumberFormatException e) {
						throw new InvalidInputDataException("无效的value:" + token, file, lineNr);
					}
					
					max_index = Math.max(max_index, x[j].index);
				}
	
				vx.add(x);
			}
			
			return constructProblem(vy, vx, max_index, bias);
		} finally {
			fp.close();
		}
	}
	
	/**
	 * 
	 * */
	public static Problem constructProblem(List<int[]> vy, List<DataPoint[]> vx, int max_index, double bias) {
		Problem prob = new Problem();
		prob.bias = bias;
		prob.l = vy.size();
		prob.n = max_index;
		
		if(bias >= 0) {
			prob.n++;
		}
		prob.x = new DataPoint[prob.l][];
		for(int i = 0; i < prob.l; i++) {
			prob.x[i] = vx.get(i);
			if(bias >= 0) {
				assert prob.x[i][prob.x[i].length - 1] == null;
				prob.x[i][prob.x[i].length - 1] = new DataPoint(max_index + 1, bias);
			}
		}
		
		prob.y = new int[prob.l][];
		for(int i = 0; i < prob.l; i++) {
			prob.y[i] = vy.get(i);
		}
		
		return prob;
	}
	
	public static int[][] getLabels(int[][] labels, Map<Integer, Integer> labelToID) {
		int[][] result = new int[labels.length][];
		
		int[] temp = null;
		int i, j;
		
		for(i = 0; i < labels.length; i++) {
			temp = labels[i];
			for(j = 0; j < temp.length; j++) {
				temp[j] = labelToID.get(temp[j]);
			}
			result[i] = temp;
		}
		return result;
	}
	
	public static Set<Integer> getIndex(String filename) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(
				new FileInputStream(filename)));
		
		String line = null;
		
		Set<Integer> set = new HashSet<Integer>();
		String[] splits = null;
		int i;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\r|\n|:");
			for(i = 1; i < splits.length / 2; i++) {
				set.add(Integer.parseInt(splits[2 * i + 1]));
			}
		}
		
		return set;
	}
	
	/**
	 *  保存prob到文件
	 * @throws IOException 
	 */
	public static void writeProbToFile(Problem prob, String filename) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		String line = null;
		for(int i = 0; i < prob.l; i++) {
			line = new String();
			for(int j = 0; j < prob.y[i].length; j++) {
				line = line + prob.y[i][j];
				if(j < prob.y[i].length - 1) {
					line = line + ",";
				} else {
					line = line + " ";
				}
			}
			
			for(int j = 0; j < prob.x[i].length; j++) {
				line = line + prob.x[i][j].index + ":" + prob.x[i][j].value;
				if(j < prob.x[i].length - 1) {
					line = line + " ";
				} else {
					line = line + "\n";
				}
			}
			out.write(line);
		}
		out.close();
	}
	
	/**
	 * probb添加到proba之后
	 */
	public static Problem mergeProblem(Problem proba, Problem probb) {
		int l = proba.l + probb.l;
		int n = Math.max(proba.n, probb.n);
		
		Problem prob = new Problem();
		prob.bias = proba.bias;
		prob.l = l;
		prob.n = n;
		prob.x = new DataPoint[prob.l][];
		prob.y = new int[prob.l][];
		
		int counter = 0;
		for(int i = 0; i < proba.l; i++) {
			prob.x[counter] = proba.x[i];
			prob.y[counter] = proba.y[i];
			counter++;
		}
		
		for(int i = 0; i < probb.l; i++) {
			prob.x[counter] = probb.x[i];
			prob.y[counter] = probb.y[i];
			counter++;
		}
		
		return prob;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static void writeObjectToFile(Object obj, String filename) throws IOException {
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
		out.writeObject(obj);
		out.close();
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static void writeSparseMat(DataPoint[][] w, String file) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(file)));
		for(int i = 0; i < w.length; i++) {
			String line = new String();
			for(int j = 0; j < w[i].length; j++) {
				line = line + w[i][j].index + ":" + w[i][j].value + " ";
			}
			line = line + "\n";
			out.write(line);
		}
		out.close();
	}
	
	/**
	 * @throws InvalidInputDataException 
	 * @throws IOException 
	 * 
	 */
	public static void writeSplitProb(String filename, int nsplit) throws IOException, InvalidInputDataException {
		Problem prob = newReadProblem(new File(filename), -1);
		int n = prob.l / nsplit;
		
		String basefilename = "splits";
		int j = 0;
		for(int i = 0; i < nsplit - 1; i++) {
			Problem nprob = new Problem();
			nprob.l = n;
			nprob.n = prob.n;
			nprob.bias = prob.bias;
			nprob.x = new DataPoint[n][];
			nprob.y = new int[n][];
			
			int counter = 0;
			for(counter = 0; counter < n; counter++) {
				nprob.x[counter] = prob.x[j];
				nprob.y[counter] = prob.y[j];
				j++;
			}
			
			String file = basefilename + i + ".txt";
			writeProbToFile(nprob, file);
		}
	System.out.println("j = " + j);	
		int counter = 0;
		int nums = prob.l - j;
		Problem nprob = new Problem();
		nprob.l = nums;
		nprob.n = prob.n;
		nprob.bias = prob.bias;
		nprob.x = new DataPoint[nums][];
		nprob.y = new int[nums][];
		for(counter = 0; counter < nums; counter++) {
			nprob.x[counter] = prob.x[j];
			nprob.y[counter] = prob.y[j];
			j++;
		}
		String file = basefilename + (nsplit - 1) + ".txt";
		writeProbToFile(nprob, file);
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static  Map<Integer, Integer> readMap(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		Set<Integer> set = new HashSet<Integer>();
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\n|\r|\t");
			for(int i = 0; i < splits.length; i++) {
				set.add(Integer.parseInt(splits[i].trim()));
			}
		}
		in.close();
		
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int id = 1;
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			map.put(it.next(), id++);
		}
		return map;
	}
	
	
	/**
	 * @throws IOException 
	 * @throws NumberFormatException 
	 * 
	 */
	public static int[] getRoots(String filename) throws NumberFormatException, IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		Set<Integer> set = new HashSet<Integer>();
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\n|\r|\t");
			for(int i = 0; i < 1; i++) {
				set.add(Integer.parseInt(splits[i].trim()));
			}
		}
		in.close();
		
		int[] roots = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		int counter = 0;
		while(it.hasNext()) {
			roots[counter++] = it.next();
		}
		return roots;
	}
	/**
	 * @throws IOException 
	 * @throws NumberFormatException 
	 * 
	 */
	public static Structure getStructure(String filename, Map<Integer, Integer> map) throws NumberFormatException, IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		int[] labels = null;
		
		int[] roots = getRoots(filename);
		Structure tree = new Structure(map.size() + 1);
		for(int i = 0; i < roots.length; i++) {
			tree.addChild(0, map.get(roots[i]));
		}
		
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\n|\r|\t");
			labels = new int[splits.length];
			for(int i = 0; i < labels.length - 1; i++) {
				int parent = Integer.parseInt(splits[i].trim());
				int child = Integer.parseInt(splits[i+1].trim());
				tree.addChild(map.get(parent), map.get(child));
			}
		}
		in.close();
		return tree;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static void readLabels(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		Set<Integer> set = new HashSet<Integer>();
		while((line = in.readLine()) != null) {
			String[] splits = line.split("\\s+|\r|\n|\t");
			for(int i = 0; i < splits.length; i++) {
				set.add(Integer.parseInt(splits[i].trim()));
			}
		}
		
		int counter = 1;
		int[] labels = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			labels[counter++] = it.next();
		}
		Arrays.sort(labels);
		for(int i = 0; i < labels.length; i++) {
			System.out.println(i+"--" + labels[i]);
		}
		in.close();
	}
	
	/**
	 * 
	 */
	public static void transLabels(int[][] y, Map<Integer, Integer> map) {
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				y[i][j] = map.get(y[i][j]);
			}
		}
	}
	
	/**
	 * DF 降维 ，获得降维之后item编号
	 */
	public static Set<Integer> getItems(Problem prob, int df) {
		Set<Integer> set = new HashSet<Integer>();
		DataPoint dp = null;
		for(int i = 0; i < prob.l; i++) {
			for(int j = 0; j < prob.x[i].length; j++) {
				dp = prob.x[i][j];
				if(dp.value > df) {
					set.add(dp.index);
				}
			}
		}
		return set;
	} 
	
	/**
	 * 返回词频大于df的index及到新样本空间的映射
	 */
	public static Map<Integer, Integer> indexToNex(Set<Integer> set) {
		int index = 1;
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			map.put(it.next(), index++);
		}
		return map;
	}
	
	/**
	 * 
	 */
	public static void transProb(Problem prob, Map<Integer, Integer> map) {
		DataPoint[] dp = null;
		DataPoint[] dpnew = null;
		int counter = 0;
		for(int i = 0; i < prob.l; i++) {
			dp = prob.x[i];
			counter = 0;
			for(int j = 0; j < dp.length; j++) {
				if(map.containsKey(dp[j].index)) {
					counter++;
				}
			}
			
			dpnew = new DataPoint[counter];
			counter = 0;
			for(int j = 0; j < dp.length; j++) {
				if(map.containsKey(dp[j].index)) {
					dpnew[counter++] = new DataPoint(map.get(dp[j]), dp[j].value);
				}
			}
			
			prob.x[i] = dpnew;
		}
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static void writeMapToFile(Map<Integer, Integer> map, String filename) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		Set<Integer> set = map.keySet();
		Iterator<Integer> it = set.iterator();
		int key = 0;
		int value = 0;
		String line = null;
		while(it.hasNext()) {
			key = it.next();
			value = map.get(key);
			line = new String();
			line = key + " " + value + "\n";
			out.write(line);
		}
		out.close();
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static Map<Integer, Integer> readMapFromFile(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		int key;
		int value;
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\n|\r|\t");
			key = Integer.parseInt(splits[0]);
			value = Integer.parseInt(splits[1]);
			map.put(key, value);
		}
		in.close();
		return map;
	}
	
	public static Map<Integer, Integer> readDMOZ2012Map(String filename) throws IOException {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		BufferedReader in = new BufferedReader(new InputStreamReader(
				new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		int id = 0;
		int key = 0;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\t|\n|\r");
			key = Integer.parseInt(splits[0]);
			if(!map.containsKey(key)) {
				map.put(key, id);
				id++;
			}
			
			key = Integer.parseInt(splits[1]);
			if(!map.containsKey(key)) {
				map.put(key, id);
				id++;
			}
		}
		in.close();
		return map;
	}

	public static Structure getDMOZ2012Structure(String filename, Map<Integer, Integer> map) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(
				new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		Structure tree = new Structure(map.size());
		int par = 0;
		int chi = 0;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+|\r|\t|\n");
			par = Integer.parseInt(splits[0]);
			chi = Integer.parseInt(splits[1]);
			tree.addChild(map.get(par), map.get(chi));
		}
		return tree;
	}
}