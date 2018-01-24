package com.refinedExpert;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.structure.Structure;

public class RefindeExpertTest {

	/**
	 * 原始样本进交叉验证变换为新样本。用于训练支持向量机的样本不会再经过该支持向量机输出预测结果。 
	 */
	@Test
	public void test() throws IOException, InvalidInputDataException {
		int[] fold = {5, 10, 15, 20, 25, 30 };
		for (int i = 0; i < fold.length; i++) {
			System.out.println("fold = " + fold[i]);
			trainTest(fold[i]);
		}
	}

	/**
	 * @throws InvalidInputDataException 
	 * @throws IOException 
	 * 
	 */
	public void trainTest(int fold) throws IOException, InvalidInputDataException {
		String filename = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_train.hf";
		String trainfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_train";
		String testfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_test";
		Map<String, Integer> map = ProcessIMCLEF.readStructure(filename);
		int[][] pc = ProcessIMCLEF.edge(filename, map);
					
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
				
		int[] leaves = tree.getLeaves();
		
		Problem train = Problem.readProblem(new File(trainfile), 1, map);
		train.y = ProcessIMCLEF.pathToLeaf(train.y, leaves);
		
		Problem test = Problem.readProblem(new File(testfile), 1, map);
		test.y = ProcessIMCLEF.pathToLeaf(test.y, leaves);
		
		Parameter param = new Parameter(1.0, 1000, 0.001);
		double[] k = {1, 3, 5, 7, 11, 15, 21, 31, 41, 51};
		double[] cs = new double[15];
		for (int i = 0; i < 15; i++) {
			cs[i] = Math.pow(2, i-7);
		}
		
		for (int i = 0; i < cs.length; i++) {
			param.setC(cs[i]);
			RefinedExpert fe = new RefinedExpert(train, param, k, fold, RefinedExpert.MICROF1);
			fe.trainAndTest(test);
		}
	}
}
