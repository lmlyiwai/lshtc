package com.emotion;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.OneVsAllMultilabel;
import com.sparseVector.DataPoint;

public class TestEmotionSVMKNN {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\emotion\\emotions-train.svm";
		String testfile = "F:\\DataSets\\emotion\\emotions-test.svm";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Parameter param = new Parameter(4, 5000, 0.001);
		OneVsAllMultilabel rs = new OneVsAllMultilabel(train, param);
		double[] c = new double[15];
		for(int i = -7; i < 8; i++) {
			c[i + 7] = Math.pow(2, i);
		}
		int[] K = {1, 3, 5, 7, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101};
		for(int i = 0; i < c.length; i++) {
			rs.gridSerach(train, param, 5, c[i], K);
		}
		

		Problem test = Problem.readProblem(new File(testfile), 1);
		
		DataPoint[][] w = rs.train(train, param);
		
		
		double[][] trainpv = rs.predictValues(w, train.x);
		rs.scale(trainpv);
		
		double[][] testpv = rs.predictValues(w, test.x);
		rs.scale(testpv);
		
		int[][] pre = rs.predictNear(trainpv, testpv, train.y, 31);
		
		double microf1 = Measures.microf1(rs.getUniqueLabels(), test.y, pre);
		double macrof1 = Measures.macrof1(rs.getUniqueLabels(), test.y, pre);
		double hammmingLoss = Measures.averageSymLoss(test.y, pre);
		double zeroneloss = Measures.zeroOneLoss(test.y, pre);
		
		System.out.println("C = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
				", Hamming Loss = " + hammmingLoss + ", Zero One Loss = " + zeroneloss);

//		String testlabel = "F:\\����\\ʵ������\\emotion\\����������ǩ.txt";
//		String prelabel = "F:\\����\\ʵ������\\emotion\\��������Ԥ���ǩSVM_KNN.txt";
//		FileIO.writeLabelToFile(testlabel, test.y);
//		FileIO.writeLabelToFile(prelabel, pre);
	}

}
