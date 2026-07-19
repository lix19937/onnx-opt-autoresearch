# onnx-opt-autoresearch  

## workflow  
```
onnx -> -------                     -> ort verify -> trtexec --fp16  latency  baseline-0       |
onnx -> onnxsim                     -> ort verify -> trtexec --fp16  latency  baseline-1       |   ->   baseline  
onnx -> polygraph surgeon sanitize  -> ort verify -> trtexec --fp16  latency  baseline-2       |

onnx -> auto opt                    -> ort verify -> trtexec --fp16  latency  exp
                                                          if exp < baseline,  { choose and flag, continue opt 5 } ;
                                                          else  enter new loop 


```

## struct   
```
references    
scripts    
SKILL.md    
```
## Optimization Techniques for Computation Graphs     
+ Constant Folding: Statically computes subgraphs that depend solely on constant nodes during the offline phase, avoiding redundant computations during inference.    
+ Redundant Node Elimination: Removes nodes that do not affect the model output (e.g., Identity, Dropout, Unsqueeze, etc.) to streamline the computation graph.    
+ Operator Fusion: Merges multiple consecutive operators into a single operator to reduce kernel launch overhead and optimize memory access (read/write).
Typical scenarios: Conv + Add, Conv + BatchNorm + ReLU, GELU fusion, LayerNorm fusion.    
+ Layout Optimization: Adjusts tensor layouts to a more efficient format for specific hardware backends (e.g., converting from NCHW to NHWC for CUDA).   

## opt-methods    
+ 常量折叠 (Constant Folding)： 在离线阶段静态计算仅依赖常量节点的子图，避免在推理运行时重复计算。    
+ 冗余节点消除 (Redundant Node Elimination)： 移除对模型输出无影响的节点（如 Identity、Dropout、Unsqueeze 等），精简计算图。    
+ 算子融合 (Node/Operator Fusion)： 将多个连续的算子合并为一个算子，以减少 Kernel Launch 开销和显存读写。    
典型场景： Conv + Add、Conv + BatchNorm + ReLU、GELU 融合、LayerNorm 融合。   
+ 布局优化 (Layout Optimization)： 针对特定硬件后端（如 CUDA），将张量布局调整为更高效的格式（如从 NCHW 转换为 NHWC）。    


## requirements       
run on SOC better       

+ python
```
onnx 
onnxruntime  
polygraph  
onnxsim              https://github.com/onnxsim/onnxsim     
onnx_graphsurgeon
onnxslim             https://github.com/inisis/OnnxSlim
```

+ bin
```
trtexec
```

## 尾声     
本skill 是基于导出后的onnx 开展的研究  

导出前的工作很重要！！！  
模型设计与导出层面的优化：固定形状、选择高版本 opset、高版本torch         

## references  
https://github.com/zzqiuzz/onnx-opt-tool/tree/master    
https://github.com/ZhongkuiMa/slimonnx      
https://github.com/levipereira/Model-Optimizer-ONNX     
https://github.com/onnx/optimizer    
https://github.com/ThanatosShinji/onnx-tool   


