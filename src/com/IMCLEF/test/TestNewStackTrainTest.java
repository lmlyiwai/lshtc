package com.IMCLEF.test;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.flatSvm.FlatSVM;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class TestNewStackTrainTest {

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
		Parameter param = new Parameter(1.0, 1000, 0.001);
		
		Problem prob = Problem.readProblem(new File(trainfile), 1, map);
		prob.y = ProcessIMCLEF.pathToLeaf(prob.y, leaves);
		
		Problem testprob = Problem.readProblem(new File(testfile), 1, map);
		testprob.y = ProcessIMCLEF.pathToLeaf(testprob.y, leaves);
		
		FlatSVM fs = new FlatSVM(prob, param);
		fs.setTree(tree);
		
		double[] c = new double[10];

		for(int i = 0; i < c.length; i++) {
			for(int j = 0; j < c.length; j++) {
				double c1 = c[i];
				double c2 = c[j];
				param.setC(c1);
				DataPoint[][] fw = fs.getFirstWeight(prob, param, tree);
				DataPoint[][] sw = fs.getSecondWeight(prob, c2, 1, tree, fw, prob.n, true, 1000);
				int[][] pl = fs.secondLevelPredictMax(sw, prob.x);		
				double microf1 = Measures.microf1(fs.getUlabels(), prob.y, pl);
				double macrof1 = Measures.macrof1(fs.getUlabels(), prob.y, pl);
				
				int[][] tpl = fs.secondLevelPredictMax(sw, testprob.x);
				double testMicrof1 = Measures.microf1(fs.getUlabels(), testprob.y, tpl);
				double testMacrof1 = Measures.macrof1(fs.getUlabels(), testprob.y, tpl);
				System.out.printf("C1 = %5.4f, C2 = %5.4f, Train, Mif1 = %5.4f, Maf1 = %5.4f, Test, Mif1 = %5.4f, Maf1 = %5.4f\n", 
						param.getC(), c2, microf1, macrof1, testMicrof1, testMacrof1);
			}
		}
	}

}
