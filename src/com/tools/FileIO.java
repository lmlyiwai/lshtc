package com.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

/**
 * 读取，写入文件等操作
 * */
public class FileIO {
	/**
	 * @throws IOException 
	 * 读取训练样本
	 * @throws InvalidInputDataException 
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
				String[] labels = label.split(",\\s|,");
				int[] labs = new int[labels.length];		//样本对应标签
				for(int i = 0; i < labs.length; i++) {
					labs[i] = Integer.parseInt(labels[i]);
				}
				vy.add(labs);
				
				int m = st.length / 2;
				DataPoint[] x;
				if(bias >= 0) {
					x = new DataPoint[m+1];
				} else {
					x = new DataPoint[m];
				}
				
				int indexBefore = 0;
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
					if(index <= indexBefore) throw new InvalidInputDataException("index 应该以递增方式排列", file, lineNr);
					indexBefore = index;
				
					token  = st[2 * j + 2];
					try {
						double value = Double.parseDouble(token);
						x[j] = new DataPoint(index, value);
					} catch (NumberFormatException e) {
						throw new InvalidInputDataException("无效的value:" + token, file, lineNr);
					}
				}
				
				if(m > 0) {
					max_index = Math.max(max_index, x[m-1].index);
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
	
	/**
	 *  标签写入文件
	 * @throws IOException 
	 */
	public static void writeLabelToFile(String filename, int[][] y) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(new File(filename))));
		String line = null;
		for(int i = 0; i < y.length; i++) {
			line = new String();
			for(int j = 0; j < y[i].length; j++) {
				line = line + y[i][j] + " ";
			}
			line = line + "\n";
			out.write(line);
		}
		out.close();
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static int[][] getLabelFromFile(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(new File(filename))));
		String line = null;
		int[] temp = null;
		String[] splits = null;
		List<int[]> list = new ArrayList<int[]>();
		while((line = in.readLine()) != null) {
			splits = line.split("\\s+");
			temp = new int[splits.length];
			for(int j = 0; j < temp.length; j++) {
				temp[j] = Integer.parseInt(splits[j]);
			}
			list.add(temp);
		}
		in.close();
		
		int[][] result = new int[list.size()][];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 * arff 格式转化为svm可读模式
	 * @throws IOException 
	 */
	public static void transArffToSVM(String arffile, String outputfile, int numOfFeatures) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(
				new FileInputStream(arffile)));
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(outputfile)));
		String line = null;
		String[] splits = null;
		String[] labels = null;
		
		String label = null;
		String feature = null;
		while((line = in.readLine()) != null) {
			if(!line.startsWith("@") && line.length() != 0) {
				splits = line.split(",");
				labels = new String[splits.length - numOfFeatures];
				for(int i = numOfFeatures; i < splits.length; i++) {
					labels[i - numOfFeatures] = splits[i];
				}
				
				label = getLabels(labels);
				feature = new String();
				for(int i = 0; i < numOfFeatures; i++) {
					feature = feature + (i + 1) + ":" + splits[i] + " ";
				}
				
				line = label + feature + "\n";
				out.write(line);
			}
		}
		
		in.close();
		out.close();
	}
	
	/**
	 *  
	 */
	public static String getLabels(String[] a) {
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < a.length; i++) {
			if(a[i].equals("1")) {
				list.add(i);
			}
		}
		
		String result = new String();
		for(int i = 0; i < list.size(); i++) {
			result = result + list.get(i);
			if(i < list.size() - 1) {
				result = result + ",";
			} else {
				result = result + " ";
			}
		}
		return result;
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static void trans(String inputfile, String outputfile) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(inputfile)));
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(outputfile)));
		String line = null;
		String[] splits = null;
		while((line = in.readLine()) != null) {
			splits = line.split(",|\\s+|\r|\n|\t");
			line = new String();
			line = splits[0] + " ";
			for(int i = 1; i < splits.length; i++) {
				line = line + i + ":" + splits[i] + " ";
			}
			line = line + "\n";
			out.write(line);
		}
		in.close();
		out.close();
	}
}
