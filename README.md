# haein

### 내가 작성한 코드
```python 

# 구현: 행 마다 가장 작은 숫자 sort (for문) -> list 에 append 해준 뒤에 가장 큰 숫자 찾기 .max()
# 예상되는 문제) 수가 커질 경우 for 및 list 에서 TLE 가능성


n, m = map(int, input().split())
num = []
for i in range(n):
    data = list(map(int, input().split()))
    num.append(min(data))

result = max(num)

print(result)

```

