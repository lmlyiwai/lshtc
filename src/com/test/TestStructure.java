package com.test;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.RecursiveSVM;
import com.structure.Structure;
import com.tools.FileInputOutput;

public class TestStructure {

	@Test
	public void test() throws Exception, InvalidInputDataException {
		Structure tree = new Structure(7);
		tree.addChild(0, 1);
		tree.addChild(0, 2);
		tree.addChild(1, 3);
		tree.addChild(1, 4);
		tree.addChild(2, 5);
		tree.addChild(2, 6);
		tree.extendStructure();
		
		String filename = "test.txt";
		Problem prob = Problem.readProblem(new File(filename), 1);
		prob.y = FileInputOutput.transLabels(prob.y, tree, tree.getInnerToAdd());
		Parameter param = new Parameter(1, 1000, 0.001);
		RecursiveSVM rs = new RecursiveSVM(tree, prob, param, 0.001);
		double[] cost = rs.getCost(prob.y, 7);
		for(int i = 0; i < cost.length; i++) {
			for(int j = 0; j < prob.y[i].length; j++) {
				System.out.print(prob.y[i][j] + " ");
			}
			System.out.println("cost = " + cost[i]);
		}
		
	}

}
