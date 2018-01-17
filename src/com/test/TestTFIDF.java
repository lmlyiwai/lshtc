package com.test;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.dmoz.ReadData;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.sparseVector.SparseVector;
import com.tools.Sigmoid;
import com.tools.TFIDF;

public class TestTFIDF {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		int[][] y = {{1}, 
				{1}, 
				{1}, 
				{2}, 
				{2},
				{3},
				{3},
				{4},
				{4},
				{3},
				{7},
				{1},
				{6}};
		long start = System.currentTimeMillis();
		for(int i = 0; i < 3488; i++) {
			int[] label = multiClass(y);
		}
		long end = System.currentTimeMillis();
		System.out.println((end - start) + "ms");
	}

	/**
	 * y其实为一列向量
	 */
	public static int[] multiClass(int[][] y) {
		int[] ys = new int[y.length];
		int[] ys_count = new int[y.length];
		int pointer = -1;
		boolean contain = false;
		
		for(int i = 0; i < y.length; i++) {
			int ty = y[i][0];
			contain = false;
			for(int j = 0; j <= pointer; j++) {
				if(ty == ys[j]) {
					ys_count[j]++;
					contain = true;
				}
			}
			
			if(!contain) {
				ys[++pointer] = ty;
				ys_count[pointer] = 1;
			}
		}
		
		int max = Integer.MIN_VALUE;
		int index = -1;
		for(int i = 0; i < pointer; i++) {
			if(ys_count[i] > max) {
				max = ys_count[i];
				index = i;
			}
		}
		
		int[] label = new int[1];
		label[0] = ys[index];
		return label;
	}
}
