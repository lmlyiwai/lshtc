package com.test;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.knn.MLKNN;
import com.rssvm.Measures;

public class TestMLKNN {

	@Test
	public void test() throws IOException, InvalidInputDataException {
//		String trainfile = "F:\\DataSets\\scene\\scene-train.svm";
//		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
		String trainfile = "F:\\DataSets\\yeast\\yeast_train.svm";
		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		
		MLKNN knn = new MLKNN(train);
		knn.getStatistic(train, 20, 1);
		
		int[][] y = knn.predict(train, test.x);
		
		double microf1 = Measures.microf1(knn.getLabels(), test.y, y);
		double macrof1 = Measures.macrof1(knn.getLabels(), test.y, y);
		double hammingloss = Measures.averageSymLoss(test.y, y);
		double zeroneloss = Measures.zeroOneLoss(test.y, y);
		System.out.println("K = " + knn.getK() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 +
				", Hamming Loss = " + hammingloss + ", 0/1 loss = " + zeroneloss);
	}

}
