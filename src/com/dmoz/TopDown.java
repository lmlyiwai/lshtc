package com.dmoz;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.structure.Structure;

public class TopDown {
	private Problem prob;
	private Structure tree;
	private Parameter param;
	private int[] ulabels;
	
	public TopDown(Problem prob, Parameter param, Structure tree) {
		this.prob = prob;
		this.param = param;
		this.tree = tree;
		this.ulabels = getuLabels(prob.y);
	}
	
	public void train(Problem prob, Parameter param, String wfile) {
		int[] leveltravel = this.tree.levelTraverse();
		
		for(int i = 0; i < leveltravel.length; i++) {
			int id = leveltravel[i];
		}
	}
	
	
	/**
	 * 
	 */
	public int[] getuLabels(int[][] y) {
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < y[i].length; j++) {
				set.add(y[i][j]);
			}
		}
		
		int[] result = new int[set.size()];
		int index = 0;
		Iterator<Integer> it = set.iterator();
		while(it.hasNext()) {
			result[index++] = it.next();
		}
		return result;
	}
}
