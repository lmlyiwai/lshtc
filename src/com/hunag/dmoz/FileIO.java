/**
 * 样本读取相关函数 
 */
package com.hunag.dmoz;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

public class FileIO {
	/**
	 * @param lines文本内容，startFromZero是否从index0开始
	 * @return 处理之后的字符串数组
	 */
	public static String[][] parseContent(String[] lines, boolean startFromZero) {
		if (lines == null) {
			return null;
		}
		String[][] content = new String[lines.length][];
		for (int i = 0; i < lines.length; i++) {
			content[i] = parseLine(lines[i], startFromZero);
		}
		return content;
	}
	
	/**
	 * @param filename样本文件
	 * @return 文件内容字符串数组
	 * @throws IOException 
	 */
	public static String[] readFile(String filename) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(
				new FileInputStream(new File(filename))));
		String line = null;
		List<String> contentList = new ArrayList<String>();
		while ((line = in.readLine()) != null) {
			contentList.add(line);
		}
		in.close();
		String[] contentArray = new String[contentList.size()];
		for (int i = 0; i < contentArray.length; i++) {
			contentArray[i] = contentList.get(i);
		}
		return contentArray;
	}
	
	/**
	 * @param line
	 * @param startFromZero样本index是否从0开始，若从零开始则需要去除index 0。
	 * @return 返回字符串数组。第一个元素为类标，可能包含以逗号分隔的多个类标。
	 * 					之后是按index排序的特征值。startFromZero为ture时去除index 0。
	 */
	private static String[] parseLine(String line, boolean startFromZero) {
		if (null == line) {
			return null;
		}
		String[] splits = line.split("\\s+|\t|\r|\n");
		String label = splits[0];
		Map<String, String> indexValueMap = new HashMap<String, String>();
		List<Integer> keyList = new ArrayList<Integer>();
		for (int i = 1; i < splits.length; i++) {
			String[] kv = splits[i].split(":");
			keyList.add(Integer.parseInt(kv[0]));
			indexValueMap.put(kv[0], kv[1]);
		}
		
		int[] index = sortIndex(keyList);
		String[] result = new String[index.length * 2 + 1];
		int inx = 1;
		int base = 0;
		if (startFromZero) {
			base = 1;
			result = new String[(index.length - 1)* 2 + 1];
		}
		result[0] = label;
		while (base < index.length) {
			String key = index[base] + "";
			String value = indexValueMap.get(key);
			result[inx++] = key;
			result[inx++] = value;
			base++;
		}
		return result;
	}
	
	/**
	 * @param indexList包含index值
	 * @return 返回排序后index值
	 */
	private static int[] sortIndex(List<Integer> indexList) {
		if (indexList == null || indexList.size() == 0) {
			return null;
		}
		int[] sortedIndex = new int[indexList.size()];
		int index = 0;
		while (indexList.size() != 0) {
			int min = Integer.MAX_VALUE;
			int ind = -1;
			for (int i = 0; i < indexList.size(); i++) {
				if (indexList.get(i) < min) {
					min = indexList.get(i);
					ind = i;
				}
			}
			sortedIndex[index++] = min;
			indexList.remove(ind);
		}
		return sortedIndex;
	}
	
	/**
	 * @throws IOException 
	 * 	
	 */
	public static void writeStringToFile(String outfile, String[][] content) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(outfile)));
		for (int i = 0; i < content.length; i++) {
			out.write(content[i][0] + " ");
			for (int j = 1; j < content[i].length; j+= 2) {
				out.write(content[i][j] + ":" + content[i][j+1] + " ");
			}
			out.write("\n");
		}
		out.close();
	}
	
	/**
	 * @param filename 文件路径
	 * @param bias偏置，bias > 0时有效
	 * @return problem
	 * @throws IOException 
	 */
	public static Problem readProblem(String filename, double bias) throws IOException {
		String[] content = readFile(filename);
		Problem prob = new Problem();
		prob.bias = bias;
		prob.l = content.length;
		prob.x = new DataPoint[prob.l][];
		prob.y = new int[prob.l][];
		
		for (int i = 0; i < content.length; i++) {
			String[] splits = content[i].split("\\s+");
			String label = splits[0];
			prob.y[i] = getLabels(label);
			prob.x[i] = getFeatures(splits, bias);
		}
		
		int maxIndex = getMaxDim(prob);
		if (bias > 0) {
			prob.n = maxIndex + 1;
			
			for (int i = 0; i < prob.l; i++) {
				int ind = prob.x[i].length - 1;
				prob.x[i][ind] = new DataPoint(maxIndex + 1, bias);
			}
		} else {
			prob.n = maxIndex;
		}
		return prob;
	}
	
	/**
	 * 
	 */
	private static int getMaxDim(Problem prob) {
		if (prob == null) {
			return -1;
		}
		int maxIndex = Integer.MIN_VALUE;
		for (int i = 0; i < prob.x.length; i++) {
			for (int j = 0; j < prob.x[i].length; j++) {
				if (prob.x[i][j] != null && prob.x[i][j].index > maxIndex) {
					maxIndex = prob.x[i][j].index;
				}
			}
		}
		return maxIndex;
	}
	
	/**
	 * 
	 */
	private static int[] getLabels(String labels) {
		if (labels == null) {
			return null;
		}
		
		String[] splits = labels.split(",|, ");
		int[] result = new int[splits.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = Integer.parseInt(splits[i]);
		}
		return result;
	}
	
	/**
	 * 
	 */
	private static DataPoint[] getFeatures(String[] splits, double bias) {
		if (splits == null) {
			return null;
		}
		
		int dim = splits.length - 1;
		if (bias > 0) {
			dim = dim + 1;
		}
		DataPoint[] dp = new DataPoint[dim];
		for (int i = 1; i < splits.length; i++) {
			String kv = splits[i];
			String[] spkv = kv.split(":");
			int index = Integer.parseInt(spkv[0]);
			double value = Double.parseDouble(spkv[1]);
			dp[i-1] = new DataPoint(index, value);
		}
		return dp;
	}
	
	/**
	 * @param probs
	 * 这里是为了处理DMOZ数据集，DMOZ数据集里index可能有冗余，
	 * 将冗余去除并为其分配新的index
	 */
	public static void getUniqueIndex(Problem[] probs) {
		if (probs == null) {
			return;
		}
		Set<Integer> indexSet = new HashSet<Integer>();
		for (int i = 0; i < probs.length; i++) {
			Problem prob = probs[i];
			for (int j = 0; j < prob.l; j++) {
				for (DataPoint dp : prob.x[j]) {
					indexSet.add(dp.index);
				}
			}
		}
		
		int index = 0;
		int[] indexs = new int[indexSet.size()];
		Iterator<Integer> it = indexSet.iterator();
		while (it.hasNext()) {
			indexs[index++] = it.next();
		}
		Arrays.sort(indexs);
		
		Map<Integer, Integer> oldToNew = new HashMap<Integer, Integer>();
		for (int i = 0; i < indexs.length; i++) {
			oldToNew.put(indexs[i], i+1);
		} 
		
		for (int i = 0; i < probs.length; i++) {
			Problem prob = probs[i];
			for (int j = 0; j < prob.l; j++) {
				for (DataPoint dp : prob.x[j]) {
					dp.index = oldToNew.get(dp.index);
				}
			}
		}
	}
	
	/**
	 * 将样本写入文件
	 * @throws IOException 
	 */
	public static void writeProbToFile(Problem prob, String filename) throws IOException {
		if (prob == null) {
			return;
		}
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		for (int i = 0; i < prob.l; i++) {
			StringBuffer sb = new StringBuffer();
			int[] label = prob.y[i];
			DataPoint[] x = prob.x[i];
			for (int j = 0; j < label.length; j++) {
				sb.append(label[j]);
				if (j == label.length - 1) {
					sb.append(" ");
				} else {
					sb.append(",");
				}
			}
			
			for (DataPoint d : x) {
				sb.append(d.index + ":" + d.value + " ");
			}
			sb.append("\n");
			out.write(sb.toString());
		}
		out.close();
	}
}
