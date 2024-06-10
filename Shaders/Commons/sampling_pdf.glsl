float valueFromSamplingPDF(vec3 origin, vec3 direction, int num_of_triangles, )
{
    float out_pdf = 0.f;
    float weight = 1.f / num_of_triangles;
    for (int i = 0; i < num_of_triangles; ++i)
    {
        out_pdf += 1.f;//calculatePDFValueOfEmittedLight(direction);
    }

    return out_pdf;
}