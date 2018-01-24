package com.dmoz;

import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.scene.TestScene;

public class Schedule {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		TestKnn tk = new TestKnn();
		tk.test();
		
		TestSecondLayer tsl = new TestSecondLayer();
		tsl.test();
		
		TestSVMKnn tsk = new TestSVMKnn();
		tsk.test();
	}

}
