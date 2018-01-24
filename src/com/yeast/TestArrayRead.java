package com.yeast;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import com.tools.FileIO;

public class TestArrayRead {

	@Test
	public void test() throws IOException {
		int[][] y = {{1,2,3},
				{2,3,4},
				{3,4,5},
				{4,5,6}};
		String filename = "testlabels.txt";
		FileIO.writeLabelToFile(filename, y);
		int[][] r = FileIO.getLabelFromFile(filename);
		for(int i = 0; i < r.length; i++) {
			for(int j = 0; j < r[i].length; j++) {
				System.out.print(r[i][j] + " ");
			}
			System.out.println();
		}
	}

}
