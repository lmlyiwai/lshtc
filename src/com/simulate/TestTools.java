package com.simulate;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;

public class TestTools {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "test.txt";
		Problem prob = Problem.readProblem(new File(filename), 1);
		int[][] index = Tools.getClassIndex(prob.y);
		for(int i = 0; i < index.length; i++) {
			for(int j = 0; j < index[i].length; j++) {
				System.out.print(index[i][j] + " ");
			}
			System.out.println();
		}
		
		int[][] tv = Tools.splits(index);
		for(int i = 0; i < tv.length; i++) {
			for(int j = 0; j < tv[i].length; j++) {
				System.out.print(tv[i][j] + " ");
			}
			System.out.println();
		}
	}

}
