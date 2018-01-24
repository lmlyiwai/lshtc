package com.tools;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class ReadArray {
	
	public static int[][] readArrayFromFile(String filename) throws IOException{
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		String[] splits = null;
		
		List<int[]> ilist = new ArrayList<int[]>();
		int[] t;
		int i;
		while((line = in.readLine()) != null) {
			if(line.length() == 0) {
				ilist.add(new int[0]);
			} else {
				splits = line.split("\\s+|\r|\t|\n");
				t = new int[splits.length];
				for(i = 0; i < t.length; i++) {
					t[i] = Integer.parseInt(splits[i]);
				}
				ilist.add(t);
			}
		}
		
		int[][] result = new int[ilist.size()][];
		for(i = 0; i < result.length; i++) {
			t = ilist.get(i);
			result[i] = t;
		}
		return result;
	}
	
	public static void showArray(int[][] array) {
		for(int i = 0; i < array.length; i++) {
			if(array[i] == null) {
				continue;
			}
			for(int j = 0; j < array[i].length; j++) {
				System.out.print(array[i][j] + " ");
			}
			System.out.println();
		}
	}
	
	public static int[] uniqueNum(int[][] array) {
		Set<Integer> set = new HashSet<Integer>();
		int i;
		int j;
		for(i = 0; i < array.length; i++) {
			for(j = 0; j < array[i].length; j++) {
				set.add(array[i][j]);
			}
		}
		
		int[] result = new int[set.size()];
		Iterator<Integer> it = set.iterator();
		i = 0;
		while(it.hasNext()) {
			result[i++] = it.next();
		}
		
		return result;
		
	}
}
