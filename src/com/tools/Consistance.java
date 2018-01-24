package com.tools;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import com.structure.Structure;

public class Consistance {
	
	public static int[][] fixLabels(Structure struc, int[][] labels) {
		
//		labels = FileInputOutput.extendLabelToOld(struc, labels);
		
		int root = struc.getRoot();
		int[][] result = new int[labels.length][];
		List<Integer> list = new ArrayList<Integer>();
		
		int i, j;
		int[] temp = null;
		int[] path;
		for(i = 0; i < labels.length; i++) {
			temp = new int[labels[i].length + 1];
			for(j = 0; j < labels[i].length; j++) {
				temp[j] = labels[i][j];
			}
			temp[temp.length - 1] = root;
			
			list.clear();
			for(j = 0; j < labels[i].length; j++) {
				path = struc.getPathToRoot(labels[i][j]);
				if(Contain.contain(temp, path)) {
					list.add(temp[j]);
				}
			}
			
			result[i] = new int[list.size()];
			for(j = 0; j < list.size(); j++) {
				result[i][j] = list.get(j);
			}
		}
		
		
//		result = FileInputOutput.transLabels(result, struc, struc.getInnerToAdd());
		
		return result;
	}
	
	/**
	 *  ×ÔµÍÏòÉÏ
	 * */
	public static int[][] bottomUp(Structure struc, int[][] labels) {
		labels = FileInputOutput.extendLabelToOld(struc, labels);
		
		int root = struc.getRoot();
		int[][] result = new int[labels.length][];
		
		Set<Integer> set = new HashSet<Integer>();
		
		int i, j, k;
		
		int[] temp = null; 
		int[] path = null;
		int counter = 0;
		for(i = 0; i < labels.length; i++) {
			temp = labels[i];
			set.clear();
			for(j = 0; j < temp.length; j++) {
				path = struc.getPathToRoot(temp[j]);
				for(k = 0; k < path.length; k++) {
					if(path[k] != root) {
						set.add(path[k]);
					}
				}
			}
			
			result[i] = new int[set.size()];
			Iterator<Integer> it = set.iterator();
			counter = 0;
			while(it.hasNext()) {
				result[i][counter++] = it.next();
			}
		}
		
		result = FileInputOutput.transLabels(result, struc, struc.getInnerToAdd());
		
		return result;
	}
	
	
	public static int[] exceptionLabels(Structure struc, int[][] labels) {
		int root = struc.getRoot();
		int[] result;
		List<Integer> list = new ArrayList<Integer>();
		
		int i, j;
		int[] temp = null;
		int pid;
		for(i = 0; i < labels.length; i++) {
			temp = labels[i];
			for(j = 0; j < temp.length; j++) {
				pid = struc.getParent(temp[j]);
				if(Contain.contain(temp, pid) || pid == root) {
				
				} else {
					list.add(i);
					break;
				}
			}
			
		}
		
		result = new int[list.size()];
		for(j = 0; j < result.length; j++) {
			result[j] = list.get(j);
		}
		
		return result;
	}
}
