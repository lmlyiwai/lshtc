package com.DMOZ2012;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import com.dmoz.ReadData;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.tools.FileInputOutput;
import com.tools.TFIDF;

public class LTC {

	public static void main(String[] args) throws IOException, InvalidInputDataException {
		// TODO 自动生成的方法存根
		String trainFile = "F:\\DataSets\\Dmoz\\DMOZ2012\\train\\track2-DMOZ-train.txt";
		String testFile = "F:\\DataSets\\Dmoz\\DMOZ2012\\test\\track2-dmoz-test.txt";
		Problem train = ReadData.newReadProblem(new File(trainFile), -1);
		Problem test = ReadData.newReadProblem(new File(testFile), -1);
		Map<Integer, Double> map = TFIDF.idf(train);
		TFIDF.ltc(train, map);
		String trainLtcFile = "F:\\DataSets\\Dmoz\\DMOZ2012\\train\\DMOZ2012_train_ltc.txt";
		ReadData.writeProbToFile(train, trainLtcFile);
		
		TFIDF.ltc(test, map);
		String testLtcFile = "F:\\DataSets\\Dmoz\\DMOZ2012\\test\\DMOZ2012_test_ltc.txt";
		ReadData.writeProbToFile(test, testLtcFile);
	}

}
