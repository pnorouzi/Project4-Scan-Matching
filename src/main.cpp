/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"

// ================
// Configuration
// ================

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
#define VISUALIZE 1
#define CPU 0
#define GPU 0

// LOOK-1.2 - change this to adjust particle count in the simulation
const float DT = 0.2f;


int N_first; 
int N_second;
vector<float> first_points;
vector<float> second_points;
glm::vec3 *dev_pos;

const float DT = 0.2f;

void readPlyfile(std::string plyfile, int* num_points, vector<float>& points) {
  std::ifstream myfile(plyfile);
  if (!myfile.is_open())
  {
    std::cout << "Error opening file: " << plyfile;
    exit(1);
  }
  std::string myString;

  if (!myfile.eof())
  {
    do {
      getline(myfile, myString);
      if (!myString.compare(0, 14, "element vertex")) {
        std::istringstream ss(myString);
        int count = 0;
        do {
          std::string temp;
          ss >> temp;
          if (count == 2)
            *num_points = std::stoi(temp);
        } while (count++ < 2);
      }
    } while (myString != "end_header");

    int i = 0;
    while (i < *num_points) {
      getline(myfile, myString);
      vector<string> tokens = utilityCore::tokenizeString(myString);
      points.push_back(glm::vec3(atof(tokens[0].c_str()),atof(tokens[1].c_str()),atof(tokens[2].c_str())))
      i++;
    }
  }
  std::cout << "Done Reading: " << plyfile << "\n";
}
/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "Project 4: Scan Matching";
  readPlyfile("bunny\\data\\bun045.ply", &N_first, first_points);
  readPlyfile("bunny\\data\\bun000.ply", &N_second, second_points);
  
  if (init(N_first, N_second, first_points, second_points)) {
    mainLoop(N_first, N_second, first_points, second_points);

#if CPU
  scanmatching::CPU::endSimulation();
#elif GPU
  scanmatching::GPU::endSimulation();
#endif

    return 0;
  }
  else {
    return 1;
  }
}
//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int N_first, int N_second, vector<float>& first_points, vector<float>& second_points) {
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
      << "Error: GPU device number is greater than the number of devices!"
      << " Perhaps a CUDA-capable GPU is not installed?"
      << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  // Window setup stuff
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    std::cout
      << "Error: Could not initialize GLFW!"
      << " Perhaps OpenGL 3.3 isn't available?"
      << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }

  // Initialize drawing state
  initVAO(N_first + N_second);

  // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
  // change the device ID.
  cudaGLSetGLDevice(0);

  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);

  // Initialize simulation
#if CPU
  scanmatching::CPU::initSimulation(N_first, N_second, first_points, second_points);
#elif GPU
  scanmatching::GPU::initSimulation(N1);
#endif

  updateCamera();

  initShaders(program);

  glEnable(GL_DEPTH_TEST);

#if CPU
  scanmatching::CPU::init(N1, N2, point);
#elif GPU
  scanmatching::GPU::init(N1);
#endif


  return true;
}

void initVAO(int N_FOR_VIS) {

  std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
  std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

  glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
  glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

  for (int i = 0; i < N_FOR_VIS; i++) {
    bodies[4 * i + 0] = 0.0f;
    bodies[4 * i + 1] = 0.0f;
    bodies[4 * i + 2] = 0.0f;
    bodies[4 * i + 3] = 1.0f;
    bindices[i] = i;
  }


  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);
  glGenBuffers(1, &boidIBO);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

  glBindVertexArray(0);
}

void initShaders(GLuint * program) {
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
    "shaders/boid.vert.glsl",
    "shaders/boid.geom.glsl",
    "shaders/boid.frag.glsl", attributeLocations, 2);
  glUseProgram(program[PROG_BOID]);

  if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
    glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
  }
  if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
    glUniform3fv(location, 1, &cameraPosition[0]);
  }
}

//====================================
// Main loop
//====================================
void runCUDA(int N_first, int N_second, vector<float>& first_points, vector<float>& second_points) {
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
  // use this buffer

  float4 *dptr = NULL;
  float *dptrVertPositions = NULL;
  float *dptrVertVelocities = NULL;

  cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
  cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

  // execute the kernel


// DO THIS
#if CPU
  ScanMatching::CPU::icp(xpoints, ypoints, N_first, N_second);
  ScanMatching::copyToDevice(N1, N2, xpoints, ypoints);
#elif GPU
  ScanMatching::GPU::icp(ScanMatching::getDevPos(), ScanMatching::getDevPos() + 3 * N1, N1, N2);
#endif
#if VISUALIZE
  ScanMatching::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
#endif
  // unmap buffer object
  cudaGLUnmapBufferObject(boidVBO_positions);
  cudaGLUnmapBufferObject(boidVBO_velocities);
}

void mainLoop(int N_first, int N_second, vector<float>& first_points, vector<float>& second_points) {
  double fps = 0;
  double timebase = 0;
  int frame = 0;
  int iter = 0;
  while (!glfwWindowShouldClose(window)) {
    std::cout << "\nIter\t" << iter++ << std::endl;
    if (iter >= 200)
      break;
    glfwPollEvents();

    frame++;
    double time = glfwGetTime();

    if (time - timebase > 1.0) {
      fps = frame / (time - timebase);
      timebase = time;
      frame = 0;
    }

    runCUDA(int N_first, int N_second, vector<float>& first_points, vector<float>& second_points);

    std::ostringstream ss;
    ss << "[";
    ss.precision(1);
    ss << std::fixed << fps;
    ss << " fps] " << deviceName;
    glfwSetWindowTitle(window, ss.str().c_str());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
    glUseProgram(program[PROG_BOID]);
    glBindVertexArray(boidVAO);
    glPointSize((GLfloat)pointSize);
    glDrawElements(GL_POINTS, N_first + N_second + 1, GL_UNSIGNED_INT, 0);
    glPointSize(1.0f);

    glUseProgram(0);
    glBindVertexArray(0);

    glfwSwapBuffers(window);
#endif
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}


void errorCallback(int error, const char *description) {
  fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (leftMousePressed) {
    // compute new camera parameters
    phi += (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
    updateCamera();
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
    updateCamera();
  }

  lastX = xpos;
  lastY = ypos;
}

void updateCamera() {
  cameraPosition.x = zoom * sin(phi) * sin(theta);
  cameraPosition.z = zoom * cos(theta);
  cameraPosition.y = zoom * cos(phi) * sin(theta);
  cameraPosition += lookAt;

  projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
  glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
  projection = projection * view;

  GLint location;

  glUseProgram(program[PROG_BOID]);
  if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
    glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
  }
}