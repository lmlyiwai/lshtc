package com.tools;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class Statistic {
	/**
	 * 
	 */
	public static String[] getUniqueLabels(int[][] y) {
		Set<String> set = new HashSet<String>();
		String line = null;
		for(int i = 0; i < y.length; i++) {
			int[] ty = y[i];
			Arrays.sort(ty);
			line = new String();
			for(int j = 0; j < ty.length; j++) {
				line += ty[j] + " ";
			}
			set.add(line);
		}
		
		Iterator<String> it = set.iterator();
		String[] result = new String[set.size()];
		int counter = 0;
		while(it.hasNext()) {
			result[counter++] = it.next();
		}
		return result;
	}
	
	/**
	 * 
	 */
	public static double[][][] getClassVector(double[][] x, int[][] y, String[] ulables) {
		int[] nums = new int[ulables.length];
		for(int i = 0; i < y.length; i++) {
			int[] ty = y[i];
			Arrays.sort(ty);
			String line = new String();
			for(int j = 0; j < ty.length; j++) {
				line += ty[j] + " ";
			}
			
			for(int j = 0; j < ulables.length; j++) {
				if(line.equals(ulables[j])) {
					nums[j]++;
					break;
				}
			}
		}
		
		double[][][] cv = new double[ulables.length][][];
		for(int i = 0; i < nums.length; i++) {
			cv[i] = new double[nums[i]][];
			nums[i] = 0;
		}
		
		for(int i = 0; i < y.length; i++) {
			int[] ty = y[i];
			Arrays.sort(ty);
			String line = new String();
			for(int j = 0; j < ty.length; j++) {
				line += ty[j] + " ";
			}
			for(int j = 0; j < ulables.length; j++) {
				if(line.equals(ulables[j])) {
					cv[j][nums[j]++] = x[i];
				}
			}
		}
		return cv;
	}
	
	public static void writeMatrixToFile(double[][] mat, String filename) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		for(int i = 0; i < mat.length; i++) {
			String line = new String();
			for(int j = 0; j < mat[i].length - 1; j++) {
				line += mat[i][j] + " ";
			}
			line += mat[i][mat[i].length - 1] + "\n";
			out.write(line);
		}
		out.close();
	}
}
