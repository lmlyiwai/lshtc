package com.stack;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;

public class TestBinarySVMDifferedtC {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\scene\\scene-train.svm";
		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
//		String trainfile = "F:\\DataSets\\yeast\\yeast_train.svm";
//		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		
		Parameter param = new Parameter(1, 1000, 0.001);
		double[] c = new double[15];
		for(int i = 0; i < 15; i++) {
			c[i] = Math.pow(2, i - 7);
		}
		
		BinarySVMDifferedtC bsvm = new BinarySVMDifferedtC(train, param);
		bsvm.allCrossValidation(train, param, 5, c);
		
		DataPoint[][] w = bsvm.train(train, param);
		double[][] testpv = bsvm.predictValues(w, test.x);
		int[][] pre = bsvm.predict(testpv);
		
		double microf1 = Measures.microf1(bsvm.getUlabels(), test.y, pre);
		double macrof1 = Measures.macrof1(bsvm.getUlabels(), test.y, pre);
		double hammingloss = Measures.averageSymLoss(test.y, pre);
		double zeroneloss = Measures.zeroOneLoss(test.y, pre);
		
		System.out.println("Micro-F1 = " + microf1 + 
				", Macro-F1 = " + macrof1 + ", Hamming Loss = " + hammingloss + 
				", 0/1 loss = " + zeroneloss);
	}

}
