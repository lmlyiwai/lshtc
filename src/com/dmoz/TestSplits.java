package com.dmoz;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;

public class TestSplits {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String file = "F:\\DataSets\\Dmoz\\dmoz_new\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_tfidf_scale.txt";
		ReadData.writeSplitProb(file, 3);
	}

}
