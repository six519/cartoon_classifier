#include <iostream>
#include <fdeep/fdeep.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <vector>

#define IMG_WIDTH 150
#define IMG_HEIGHT 100
#define MODEL_FILE "model.json"
#define TITLE "Cartoon Classifier"

using namespace std;
using namespace fdeep;
using namespace fdeep::internal;
using namespace cv;

const vector<string> cartoon_titles = {
	"Adventure Time",
	"Pokemon",
	"Sponge Bob",
	"Tom and Jerry",
};

void error_message()
{
	cout << "Invalid image or empty argument..." << endl;
}

int main(int argc, char** argv)
{

	if (argc < 2)
	{
		error_message();
		return EXIT_FAILURE;
	}

	model fmodel = load_model(MODEL_FILE, true, dev_null_logger);
	Mat image = imread(argv[1], IMREAD_COLOR);

	if (image.empty())
	{
		error_message();
		return EXIT_FAILURE;
	}

	Mat resized_image;
	resize(image, resized_image, Size(IMG_WIDTH, IMG_HEIGHT));

	tensor data = tensor_from_bytes(
		resized_image.ptr(),
		static_cast<size_t>(resized_image.rows),
		static_cast<size_t>(resized_image.cols),
		static_cast<size_t>(resized_image.channels()),
		0.0f, 
		1.0f
	); 

	tensors result = fmodel.predict({data});
	shared_float_vec vec = result[0].as_vector();

	int curr_max_index = 0;
	for(int x=0; x < vec->size(); x++)
	{
		if (vec->data()[x] > vec->data()[curr_max_index])
		{
			curr_max_index = x;
		}
	}

	rectangle(image, Point(15, 5), Point(255, 70), Scalar(0, 0, 0), -1, LINE_8);
	putText(image, "Detected Cartoon:",Point(20,30),FONT_HERSHEY_DUPLEX,0.8,Scalar(255,255,153),2,false);
	putText(image, cartoon_titles[curr_max_index].c_str(),Point(20,60),FONT_HERSHEY_DUPLEX,0.8,Scalar(255,255,153),2,false);
	imshow(TITLE, image);
	waitKey();

	return EXIT_SUCCESS;
}