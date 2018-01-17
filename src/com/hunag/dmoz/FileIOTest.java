package com.hunag.dmoz;

import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.Problem;

public class FileIOTest {

	@Test
	public void test() throws IOException {
//		String filename = "F:\\DataSets\\Dmoz\\dmoz2010_large\\dry-run_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData"
//				+ "\\validation.txt";
//		String valid = "F:\\DataSets\\dmoz2010\\validation.txt";
//		String[] content = FileIO.readFile(filename);
//		String[][] lines = FileIO.parseContent(content, true);
//		FileIO.writeStringToFile(valid, lines);
		
		String filename = "F:\\DataSets\\dmoz2010\\validation.txt";
		Problem prob = FileIO.readProblem(filename,  -1);
		System.out.println(prob.l + " " + prob.n + " " + prob.bias);
	}

}
