package com.test;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.EqualUnitedSVM;
import com.structure.Structure;

public class TestEqualUnitedSVM {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		Structure tree = new Structure(7);
		tree.addChild(0, 1);
		tree.addChild(0, 2);
		tree.addChild(1, 3);
		tree.addChild(1, 4);
		tree.addChild(2, 5);
		tree.addChild(2, 6);
		
		Parameter param = new Parameter(1, 1000, 0.001);
		
		String filename = "test.txt";
		Problem prob = Problem.readProblem(new File(filename), 1);
		
		EqualUnitedSVM us = new EqualUnitedSVM(tree, param, prob, 100, 0.001);
		
		us.train(prob, param);
		
		int[][] pre = us.predict(prob.x);
		
		for(int i = 0; i < prob.y.length; i++) {
			for(int j = 0; j < prob.y[i].length; j++) {
				System.out.print(prob.y[i][j] + " ");
			}
			System.out.print("-- ");
			for(int j = 0; j < pre[i].length; j++) {
				System.out.print(pre[i][j] + " ");
			}
			System.out.println();
		}
	}

}
