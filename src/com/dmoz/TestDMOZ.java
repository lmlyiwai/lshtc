package com.dmoz;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.tools.TFIDF;

public class TestDMOZ {

	@Test
	public void test() throws IOException, InvalidInputDataException, ClassNotFoundException {
		String trainfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train.txt";
		String trainoutfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_ltc_scale.txt";
		
		String validfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation.txt";
		String validoutfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation_ltc_scale.txt";
		
		String testfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\test.txt";
		String testoutfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\test_ltc_scale.txt";
		
		String idffile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\idf.obj";
		
		long start = System.currentTimeMillis();
		Problem train = ReadData.readProblem(new File(trainfile), -1);
		Problem valid = ReadData.readProblem(new File(validfile), -1);
		Problem test = ReadData.newReadProblem(new File(testfile), -1);
		Problem prob = ReadData.mergeProblem(train, valid);
		long end = System.currentTimeMillis();
		System.out.println("Read data " + (end - start) + "ms");
		
		start = System.currentTimeMillis();
		Map<Integer, Double> map = TFIDF.idf(prob);
		TFIDF.ltc(train, map);
		TFIDF.ltc(valid, map);
		TFIDF.ltc(test, map);
//		ObjectInputStream in = new ObjectInputStream(new FileInputStream(idffile));
//		Map<Integer, Double> map = (HashMap<Integer, Double>)in.readObject();
//		ReadData.writeObjectToFile(map, idffile);
//		TFIDF.tfidf(test, map);
		end = System.currentTimeMillis();
		System.out.println("TFIDF " + (end - start) + "ms");
		
		start = System.currentTimeMillis();
//		TFIDF.scale(prob);
//		TFIDF.scale(train);
//		TFIDF.scale(valid);
//		TFIDF.scale(test);
		end = System.currentTimeMillis();
		System.out.println("Scale " + (end - start) + "ms");
		
		start = System.currentTimeMillis();
		ReadData.writeProbToFile(train, trainoutfile);
		ReadData.writeProbToFile(valid, validoutfile);
		ReadData.writeProbToFile(test, testoutfile);
		end = System.currentTimeMillis();
		System.out.println("Write to file " + (end - start) + "ms");
	}

}
