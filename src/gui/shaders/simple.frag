#version 330

uniform vec3 view_pos;
uniform bool perspective;

in vec3 v_color;
in vec3 v_world_pos;
in vec3 v_cam_pos;

layout(location=0) out vec4 f_color;
layout(location=1) out vec4 f_world_pos_and_depth;

void main() {
    float depth = v_cam_pos.z;
    if (perspective){
        depth = length(view_pos - v_world_pos);
    }

    f_color = vec4(v_color, 1.0);
    f_world_pos_and_depth = vec4(v_world_pos, depth);
}