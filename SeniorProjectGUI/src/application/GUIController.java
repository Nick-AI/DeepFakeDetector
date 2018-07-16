package application;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import java.io.*;

public class GUIController {
	
	@FXML private TextField vidDirBox;
	@FXML private TextField trainVidBox;
	@FXML private Label trainingLabel;
	@FXML private Label trainModelLabel;
	@FXML private Button trainButton;
	@FXML private TextField desDirBox;
	@FXML private Button analyzeButton;
	
	public void analyzeVid(ActionEvent e){
		String vidDir = this.vidDirBox.getText();

		String destDir = this.desDirBox.getText();
		String nnSolverScript = "./Application.py";
		String[] cmd = {"python", nnSolverScript, vidDir, destDir};
		this.trainingLabel.setVisible(true);
		this.trainingLabel.setText("Analyzing Video...");
		ProcessBuilder pb = new ProcessBuilder(cmd[0], cmd[1], cmd[2], cmd[3]);
        File log = new File("./log");
        pb.redirectErrorStream(true);
        pb.redirectOutput(ProcessBuilder.Redirect.appendTo(log));
		try {
			Process pr = pb.start();
            pr.waitFor();
		}
		catch (IOException | InterruptedException e1) {
			e1.printStackTrace();
		}
		this.trainingLabel.setText("Done. Check destination.");
        System.out.println(vidDir);
        System.out.println(destDir);

	}
}
