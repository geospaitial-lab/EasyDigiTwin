#version 330

uniform mat4 w2c;
uniform mat4 projection;
uniform vec3 offset = vec3(0, 0, 0);

in vec3 in_vert;
in vec3 in_color;
in mat4 in_pose;

out vec3 v_color;
out vec3 v_world_pos;
out vec3 v_cam_pos;

void main() {

    vec4 world_pos = in_pose * vec4(in_vert + offset, 1.0);
    vec4 cam_pos = w2c * world_pos;
    gl_Position = projection * cam_pos;
    v_color = in_color;
    v_world_pos = world_pos.xyz;
    v_cam_pos = cam_pos.xyz;
}