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
onnxsim
onnx_graphsurgeon
```

+ bin
```
trtexec
```

## 尾声     
本skill 是基于导出后的onnx 开展的研究  

导出前的工作很重要！！！  
模型设计与导出层面的优化：固定形状、选择高版本 opset、高版本torch         
