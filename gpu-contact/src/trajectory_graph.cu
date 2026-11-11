// Keep v5's graph bridge intact. Build three conditional levels in one root.
#include "device_graph.cu"
API int dc_trajectory_upload(void*exec,void*stream){return cudaGraphUpload((cudaGraphExec_t)exec,(cudaStream_t)stream);}
__global__ void trajectory_condition(cudaGraphConditionalHandle h,const int*r){
 if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(h,r[1]<r[0]&&!r[2]);
}
static cudaError_t add_check(cudaGraph_t g,cudaGraphNode_t after,cudaGraphConditionalHandle h,
 int*state,int limit,cudaGraphNode_t*out){
 int done=4,error=5,count=6;void*args[]={&h,&state,&done,&error,&count,&limit};
 cudaKernelNodeParams p={};p.func=(void*)set_loop_condition;p.gridDim=dim3(1);p.blockDim=dim3(1);p.kernelParams=args;
 return cudaGraphAddKernelNode(out,g,&after,1,&p);
}
API int dc_trajectory(void**parts,int*outer,int*inner,int*run,void**outgraph,void**outexec){
 cudaGraph_t root=nullptr;cudaGraphConditionalHandle th,oh,ih;cudaGraphNode_t time_node;
 cudaError_t e=cudaGraphCreate(&root,0);if(e)return e;
 e=cudaGraphConditionalHandleCreate(&th,root,1,cudaGraphCondAssignDefault);if(e)goto fail;
 e=cudaGraphConditionalHandleCreate(&oh,root,1,cudaGraphCondAssignDefault);if(e)goto fail;
 e=cudaGraphConditionalHandleCreate(&ih,root,1,cudaGraphCondAssignDefault);if(e)goto fail;
 {
  cudaGraphNodeParams tp={};tp.type=cudaGraphNodeTypeConditional;tp.conditional.handle=th;tp.conditional.type=cudaGraphCondTypeWhile;tp.conditional.size=1;
  e=cudaGraphAddNode(&time_node,root,nullptr,0,&tp);if(e)goto fail;
  cudaGraph_t tb=tp.conditional.phGraph_out[0];cudaGraphNode_t init,ocheck,outer_node,finish,record,tc;
  e=cudaGraphAddChildGraphNode(&init,tb,nullptr,0,(cudaGraph_t)parts[0]);if(e)goto fail;
  // Handles persist across time steps; reset the outer condition explicitly.
  e=add_check(tb,init,oh,outer,5000,&ocheck);if(e)goto fail;
  cudaGraphNodeParams op={};op.type=cudaGraphNodeTypeConditional;op.conditional.handle=oh;op.conditional.type=cudaGraphCondTypeWhile;op.conditional.size=1;
  e=cudaGraphAddNode(&outer_node,tb,&ocheck,1,&op);if(e)goto fail;
  cudaGraph_t ob=op.conditional.phGraph_out[0];cudaGraphNode_t pre,icheck,inner_node,post,next;
  e=cudaGraphAddChildGraphNode(&pre,ob,nullptr,0,(cudaGraph_t)parts[1]);if(e)goto fail;
  e=add_check(ob,pre,ih,inner,4096,&icheck);if(e)goto fail;
  cudaGraphNodeParams ip={};ip.type=cudaGraphNodeTypeConditional;ip.conditional.handle=ih;ip.conditional.type=cudaGraphCondTypeWhile;ip.conditional.size=1;
  e=cudaGraphAddNode(&inner_node,ob,&icheck,1,&ip);if(e)goto fail;
  cudaGraph_t ib=ip.conditional.phGraph_out[0];cudaGraphNode_t solve,inext;
  e=cudaGraphAddChildGraphNode(&solve,ib,nullptr,0,(cudaGraph_t)parts[2]);if(e)goto fail;
  e=add_check(ib,solve,ih,inner,4096,&inext);if(e)goto fail;
  e=cudaGraphAddChildGraphNode(&post,ob,&inner_node,1,(cudaGraph_t)parts[3]);if(e)goto fail;
  e=add_check(ob,post,oh,outer,5000,&next);if(e)goto fail;
  e=cudaGraphAddChildGraphNode(&finish,tb,&outer_node,1,(cudaGraph_t)parts[4]);if(e)goto fail;
  e=cudaGraphAddChildGraphNode(&record,tb,&finish,1,(cudaGraph_t)parts[5]);if(e)goto fail;
  void*args[]={&th,&run};cudaKernelNodeParams kp={};kp.func=(void*)trajectory_condition;kp.gridDim=dim3(1);kp.blockDim=dim3(1);kp.kernelParams=args;
  e=cudaGraphAddKernelNode(&tc,tb,&record,1,&kp);if(e)goto fail;
 }
 e=cudaGraphInstantiate((cudaGraphExec_t*)outexec,root,0ULL);if(e)goto fail;
 *outgraph=root;return 0;
fail:cudaGraphDestroy(root);return e;
}
