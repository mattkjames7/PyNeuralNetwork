#include "cliplog.h"

float cliplog(float x, float min) {
	/* This function will clip values of x before passing to the log
	 * function so that there are not -inf's,
	 * a good value for min is 1e-40*/
	return log(max(x,min));
}
