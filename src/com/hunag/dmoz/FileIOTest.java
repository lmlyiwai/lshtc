package com.hunag.dmoz;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.Problem;

public class FileIOTest {

	@Test
	public void test() throws IOException {
		String trainfile = "F:\\DataSets\\dmoz2010\\train.txt";
		String testfile = "F:\\DataSets\\dmoz2010\\test.txt";
		String validfile = "F:\\DataSets\\dmoz2010\\validation.txt";
		
		Problem train = FileIO.readProblem(trainfile, -1);
		Problem test = FileIO.readProblem(testfile, -1);
		Problem valid = FileIO.readProblem(validfile, -1);
		
		Problem[] probs = {train, test, valid};
		String ntrainfile = "F:\\DataSets\\dmoz2010\\newDmoz\\train.txt";
		String ntestfile = "F:\\DataSets\\dmoz2010\\newDmoz\\test.txt";
		String nvalidfile = "F:\\DataSets\\dmoz2010\\newDmoz\\validation.txt";
		FileIO.writeProbToFile(train, ntrainfile);
		FileIO.writeProbToFile(test, ntestfile);
		FileIO.writeProbToFile(valid, nvalidfile);
	}

}
