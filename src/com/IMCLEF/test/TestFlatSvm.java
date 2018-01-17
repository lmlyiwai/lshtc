package com.IMCLEF.test;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.dmoz.ReadData;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.OneVsAllMultilabel;
import com.sparseVector.DataPoint;
import com.structure.Structure;
import com.tools.Sigmoid;
import com.tools.Sort;

public class TestFlatSvm {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a_train.hf";
		String trainfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a_train";
		String testfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a_test";
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
		
		double[] c = new double[15];
		for(int i = -7; i < 8; i++) {
			c[i + 7] = Math.pow(2, i);
		}

//		ReadData.writeProbToFile(prob, "CLEF_train.txt");
		
		OneVsAllMultilabel rs = new OneVsAllMultilabel(prob, param);
		for(int i = 0; i < 15; i++) {
			double tc = c[i];
			param.setC(tc);
//			System.out.println("c = " + param.getC());
//			rs.crossValidation(prob, param, 5);
		}
		
		param.setC(1);
		DataPoint[][] w = rs.train(prob, param);
		double[][] trainpv = rs.predictValues(w, prob.x);
		Sigmoid.sigmoid(trainpv, 1);
//		Sigmoid.scale(trainpv);
		Problem testprob = Problem.readProblem(new File(testfile), 1, map);
		testprob.y = ProcessIMCLEF.pathToLeaf(testprob.y, leaves);
		
//		ReadData.writeProbToFile(testprob, "CLEF_test.txt");
		double[][] testpv = rs.predictValues(w, testprob.x);

//		int[][] pre = rs.predict(testpv);
		double[] ent = Sigmoid.entropy(testpv);
		Sigmoid.sigmoid(testpv, 1);
//		Sigmoid.scale(testpv);
		int[] index = Sort.getIndexBeforeSort(ent);
		
		
		double cut = entropyThreshold(ent, 100);
		System.out.println("cut = " + cut);
		int[][] pre = rs.predictMax(testpv);
		
		for(int i = 0; i < index.length; i++) {
			int flag = 0;
			if(pre[index[i]][0] == testprob.y[index[i]][0]) {
				flag = 1;
			}
			System.out.println(flag + " - " + ent[index[i]]);
		}
		
		int[][] npre = new int[testprob.l][1];
		for(int i = 0; i < pre.length; i++) {
			if(ent[i] < cut) {
				npre[i] = pre[i];
			} else {
				npre[i] = rs.predictNear(trainpv, testpv[i], prob.y, 5);
			}
		}
		
		for(int i = 0; i < index.length; i++) {
			int flag = 0;
			if(npre[index[i]][0] == testprob.y[index[i]][0]) {
				flag = 1;
			}
			System.out.println(flag + " - " + ent[index[i]]);
		}
		
		for(int i = 0; i < pre.length; i++) {
			System.out.println(pre[index[i]][0] + " - " + npre[index[i]][0] + " - " + testprob.y[index[i]][0]);
		}
		
		double microf1 = Measures.microf1(rs.getUniqueLabels(), testprob.y, npre);
		double macrof1 = Measures.macrof1(rs.getUniqueLabels(), testprob.y, npre);
		double hammmingLoss = Measures.averageSymLoss(testprob.y, npre);
		double zeroneloss = Measures.zeroOneLoss(testprob.y, npre);
		
		System.out.println("C = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
				", Hamming Loss = " + hammmingLoss + ", Zero One Loss = " + zeroneloss);
	}

	public static double entropyThreshold(double[] ent, int num) {
		int[] index = Sort.getIndexBeforeSort(ent);
		double cut = 0;
		if(ent.length <= num) {
			return cut;
		}
		
		int ind = ent.length - 1 - num;
		return ent[index[ind]];
	}
}
