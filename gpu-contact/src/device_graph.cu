#include <cuda_runtime.h>
#include <stdint.h>
#define API extern "C" __declspec(dllexport)

__global__ void set_loop_condition(cudaGraphConditionalHandle handle,int*state,int done,int error,int count,int limit){
    if(threadIdx.x||blockIdx.x)return;
    if(state[count]>=limit&&!state[done]&&!state[error])state[error]=97;
    cudaGraphSetConditional(handle,(!state[done]&&!state[error])?1:0);
}

API const char* dc_error(int code){return cudaGetErrorString((cudaError_t)code);}
API int dc_begin(void*stream){return cudaStreamBeginCapture((cudaStream_t)stream,cudaStreamCaptureModeThreadLocal);}
API int dc_end(void*stream,void**graph){return cudaStreamEndCapture((cudaStream_t)stream,(cudaGraph_t*)graph);}
API int dc_launch(void*exec,void*stream){return cudaGraphLaunch((cudaGraphExec_t)exec,(cudaStream_t)stream);}
API int dc_destroy(void*graph,void*exec){
    cudaError_t e=cudaSuccess;if(exec)e=cudaGraphExecDestroy((cudaGraphExec_t)exec);
    if(graph){auto g=cudaGraphDestroy((cudaGraph_t)graph);if(e==cudaSuccess)e=g;}return e;
}
API int dc_instantiate(void*graph,void**exec){return cudaGraphInstantiate((cudaGraphExec_t*)exec,(cudaGraph_t)graph,0ULL);}
API int dc_while(void*child,int*state,int done,int error,int count,int limit,void**outgraph,void**outexec){
    cudaGraph_t graph=nullptr;cudaGraphConditionalHandle handle;cudaGraphNode_t loop,node,check;
    cudaError_t e=cudaGraphCreate(&graph,0);if(e)return e;
    e=cudaGraphConditionalHandleCreate(&handle,graph,1,cudaGraphCondAssignDefault);if(e)goto fail;
    {
        cudaGraphNodeParams p={};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=handle;
        p.conditional.type=cudaGraphCondTypeWhile;p.conditional.size=1;
        e=cudaGraphAddNode(&loop,graph,nullptr,0,&p);if(e)goto fail;
        cudaGraph_t body=p.conditional.phGraph_out[0];
        e=cudaGraphAddChildGraphNode(&node,body,nullptr,0,(cudaGraph_t)child);if(e)goto fail;
        void*args[]={&handle,&state,&done,&error,&count,&limit};
        cudaKernelNodeParams kp={};kp.func=(void*)set_loop_condition;kp.gridDim=dim3(1);kp.blockDim=dim3(1);kp.kernelParams=args;
        e=cudaGraphAddKernelNode(&check,body,&node,1,&kp);if(e)goto fail;
    }
    e=cudaGraphInstantiate((cudaGraphExec_t*)outexec,graph,0ULL);if(e)goto fail;
    *outgraph=graph;return 0;
fail:
    cudaGraphDestroy(graph);return e;
}

API int dc_nested(void**parts,int*outer,int*inner,int outer_limit,int inner_limit,void**outgraph,void**outexec){
    cudaGraph_t graph=nullptr;cudaGraphConditionalHandle oh,ih;cudaGraphNode_t init,outer_node,pre,icheck,inode,post,ocheck,end;
    cudaError_t e=cudaGraphCreate(&graph,0);if(e)return e;
    e=cudaGraphConditionalHandleCreate(&oh,graph,1,cudaGraphCondAssignDefault);if(e)goto fail;
    e=cudaGraphConditionalHandleCreate(&ih,graph,1,cudaGraphCondAssignDefault);if(e)goto fail;
    e=cudaGraphAddChildGraphNode(&init,graph,nullptr,0,(cudaGraph_t)parts[0]);if(e)goto fail;
    {
        cudaGraphNodeParams p={};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=oh;
        p.conditional.type=cudaGraphCondTypeWhile;p.conditional.size=1;
        e=cudaGraphAddNode(&outer_node,graph,&init,1,&p);if(e)goto fail;
        cudaGraph_t body=p.conditional.phGraph_out[0];
        e=cudaGraphAddChildGraphNode(&pre,body,nullptr,0,(cudaGraph_t)parts[1]);if(e)goto fail;
        int done=4,error=5,count=6;void*iargs[]={&ih,&inner,&done,&error,&count,&inner_limit};
        cudaKernelNodeParams ik={};ik.func=(void*)set_loop_condition;ik.gridDim=dim3(1);ik.blockDim=dim3(1);ik.kernelParams=iargs;
        e=cudaGraphAddKernelNode(&icheck,body,&pre,1,&ik);if(e)goto fail;
        cudaGraphNodeParams q={};q.type=cudaGraphNodeTypeConditional;q.conditional.handle=ih;
        q.conditional.type=cudaGraphCondTypeWhile;q.conditional.size=1;
        e=cudaGraphAddNode(&inode,body,&icheck,1,&q);if(e)goto fail;
        cudaGraph_t inner_body=q.conditional.phGraph_out[0];cudaGraphNode_t advance,inner_check;
        e=cudaGraphAddChildGraphNode(&advance,inner_body,nullptr,0,(cudaGraph_t)parts[2]);if(e)goto fail;
        e=cudaGraphAddKernelNode(&inner_check,inner_body,&advance,1,&ik);if(e)goto fail;
        e=cudaGraphAddChildGraphNode(&post,body,&inode,1,(cudaGraph_t)parts[3]);if(e)goto fail;
        void*oargs[]={&oh,&outer,&done,&error,&count,&outer_limit};
        cudaKernelNodeParams ok={};ok.func=(void*)set_loop_condition;ok.gridDim=dim3(1);ok.blockDim=dim3(1);ok.kernelParams=oargs;
        e=cudaGraphAddKernelNode(&ocheck,body,&post,1,&ok);if(e)goto fail;
    }
    e=cudaGraphAddChildGraphNode(&end,graph,&outer_node,1,(cudaGraph_t)parts[4]);if(e)goto fail;
    e=cudaGraphInstantiate((cudaGraphExec_t*)outexec,graph,0ULL);if(e)goto fail;
    *outgraph=graph;return 0;
fail:
    cudaGraphDestroy(graph);return e;
}
