package com.dmoz;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;

public class TestWriteW {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String file = "test.txt";
		String outfile = "testweight.txt";
		Problem prob = Problem.readProblem(new File(file), 1);
		ReadData.writeSparseMat(prob.x, outfile);
	}

}
