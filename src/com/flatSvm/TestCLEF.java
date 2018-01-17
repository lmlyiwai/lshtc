package com.flatSvm;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.OneVsAllMultilabel;
import com.sparseVector.DataPoint;
import com.structure.Structure;
import com.tools.Sigmoid;
import com.tools.Sort;

public class TestCLEF {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_train.hf";
		String trainfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_train";
		String testfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_test";
		Map<String, Integer> map = ProcessIMCLEF.readStructure(filename);
		int[][] pc = ProcessIMCLEF.edge(filename, map);
				
		Map<Integer, String> idTostr = ProcessIMCLEF.reverseMap(map);
		
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
				
		int[] leaves = tree.getLeaves();
		Parameter param = new Parameter(1, 2000, 0.001);
		
		Problem prob = Problem.readProblem(new File(trainfile), 1, map);
		prob.y = ProcessIMCLEF.pathToLeaf(prob.y, leaves);
		
		double[] c = new double[10];
		for(int i = -4; i < 6; i++) {
			c[i + 4] = Math.pow(2, i);
		}		

		int[] k = {3, 5, 7, 9, 11, 21, 31, 41, 51};
		
		SVMKNN sk = new SVMKNN(prob, param);
		sk.setTree(tree);
		
		for(int i = 0; i < c.length; i++) {
			param.setC(c[i]);
			for(int j = 0; j < k.length; j++) {
				System.out.print("c = " + param.getC() + ", k = " + k[j]);
				sk.crossValidation(prob, param, 5, k[j]);
			}
		}

	}

}
