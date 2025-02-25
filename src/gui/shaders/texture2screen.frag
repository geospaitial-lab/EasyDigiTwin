#version 330

uniform sampler2D tex;

in vec2 uv;

out vec4 f_value;

void main(){

    vec4 color = texture(tex, uv);

    f_value = vec4(color.xyz, 1.0);
//    f_value = vec4(uv, 0, 0);
}