package com.tools;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

public class TestITF {

	@Test
	public void test() throws IOException {
		String train_input_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\train.txt";
		String train_itf_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\train_itf.txt";
		String train_itf_norm_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\train_itf_norm.txt";
		
		String valid_input_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\validation.txt";
		String valid_itf_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\validation_itf.txt";
		String valid_itf_norm_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\valid_itf_norm.txt";
		
		String test_input_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\test.txt";
		String test_itf_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\test_itf.txt";
		String test_itf_norm_file = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\"
				+ "Task1_Train_CrawlData_Test_CrawlData\\test_itf_norm.txt";
		
//		ITF.itf(train_input_file, train_itf_file, true);
//		ITF.itf(valid_input_file, valid_itf_file, true);
//		ITF.itf(test_input_file, test_itf_file, false);
		
		ITF.itfNorm(train_input_file, train_itf_norm_file, true);
		ITF.itfNorm(valid_input_file, valid_itf_norm_file, true);
		ITF.itfNorm(test_input_file, test_itf_norm_file, false);
	}

}
